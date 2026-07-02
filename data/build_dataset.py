#!/usr/bin/env python3
"""
BanglaBridge dataset builder
============================
Generates an ORIGINAL, redistribution-safe instruction dataset for code-mixed,
romanized Bengali ("Banglish") — the register 100M+ people actually type
(e.g. "kal ki plan? ami free achi").

Every pair here is authored by us or produced by safe slot-substitution over our
own templates. Nothing is scraped. => safe to release on Hugging Face + Kaggle.

Design
------
- input   : Banglish (romanized code-mix), majority; plus a native-script and a
            mixed-script minority for register robustness.
- output  : replies IN THE USER'S REGISTER by default (the thing off-the-shelf
            models fail at) — except translate/explain tasks where the target
            language is explicit.
- schema  : JSONL with rich fields for analysis + a flat CSV for the Adaption
            Adaptive-Data pipeline (columns: original_prompt, response).

Run
---
    python data/build_dataset.py                 # build everything, default sizes
    python data/build_dataset.py --stats-only    # just print the distribution
    python data/build_dataset.py --augment       # (stub) LLM paraphrase scaling

Pure stdlib. Deterministic (seeded). No network unless --augment is wired.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# GOLD — hand-authored, genuinely natural Banglish instruction pairs.
# tuple = (instruction, output, task_type, domain, script)
# script: "romanized" | "native" | "mixed"
# ---------------------------------------------------------------------------
def _g(i, o, t, d, s="romanized"):
    return {"instruction": i.strip(), "output": o.strip(),
            "task_type": t, "domain": d, "script": s}


GOLD = [
    _g(
        "ami aj khub down feel korchi, kichu motivational bol na",
        """Ei kharap somoy ta permanent na, temporary. Ekhon ekta chhoto kaj kor —
ekTa chai bana, 10 minute hnete aay. Nijeke ekbar bol: "ami parbo".
Kal abar notun kore start korbi. Tui ekla nos, ami achi. EkTu ekTu kore hobei. 💪""",
        "emotional_support", "daily_life",
    ),
    _g(
        "ei message ta professional kore de: 'boss aj ami aste parbo na, shorir kharap'",
        """Ekta professional version:

"Good morning Sir, ajke shorir ekTu kharap thakay ami office e ashte parchhi na.
Aj ami sick leave niye nichhi. Kono urgent kaj thakle mail/phone e ami available
achi. Dhonnobad."
""",
        "rewrite", "work",
    ),
    _g(
        "ei ta English e translate kore de: 'kal amader exam ache, tai aj raat jege porbo'",
        "We have an exam tomorrow, so I'll stay up late tonight to study.",
        "translate", "education",
    ),
    _g(
        "'I'll call you later' — eta Banglish e casually kivabe bolbo?",
        "Casually: \"pore call korchi\" othoba \"porE phone korchi tomake\".",
        "translate", "daily_life",
    ),
    _g(
        "ei paragraph ta 2 line e summarize kor: 'ajke office e notun project er meeting chilo. "
        "boss bollo deadline khub tight, tai sobaike weekend eo kaj korte hote pare. team ektu tension e ache.'",
        """1. Notun project er deadline khub tight — weekend eo kaj korte hote pare.
2. Sei niye team er modhye ekTu tension cholche.""",
        "summarize", "work",
    ),
    _g(
        "dim bhaji banate ki ki lagbe ar kivabe korbo, short e bol",
        """Lagbe: 2 ta dim, ekTu piyaj kuchi, kacha lonka, nun, halud, tel.

Kivabe:
1. Dim gulo bhalo kore fetao, tate nun+halud die de.
2. Kadai te tel gorom kore piyaj-lonka bhaje.
3. Fetano dim dhele de, aste aste narte thak jotokkhon na jome jay. Hoye gelo! 🍳""",
        "how_to", "food",
    ),
    _g(
        "Bharoter rajdhani kothay?",
        "Bharoter rajdhani New Delhi.",
        "qa", "education",
    ),
    _g(
        "python e ekta list ke reverse korbo kivabe?",
        """Duita sohoj upay:
- `my_list[::-1]`  -> notun reversed list return kore.
- `my_list.reverse()`  -> in-place reverse kore (original tai bodle jay).

Example: `[1, 2, 3][::-1]` -> `[3, 2, 1]`.""",
        "code_help", "tech",
    ),
    _g(
        "ei review ta positive na negative bol: 'product ta valo but delivery khub late chilo'",
        """Mishro (mixed), tobe sadharon bhabe ekTu negative dike.
- Positive: product er quality niye.
- Negative: delivery late howa niye.""",
        "classification", "tech",
    ),
    _g(
        "amar coffee r photo r jonno ekta short Instagram caption de Banglish e",
        """Kichu option:
1. "Coffee ache, tai mon o bhalo ache ☕"
2. "Ek cup e reset ✨"
3. "Kaaj ache, kintu age coffee ☕📖\"""",
        "creative", "food",
    ),
    _g(
        "ami student, mase 2000 taka bachate chai, kivabe? practical tips de",
        """Realistic plan:
1. Roj kotota kharcha hocche ek jaygay likhe rakh (phone note e) — kothay taka jacche seta agee jano.
2. Baire khawa 50% komiye de — mase eitatei onek taka bachbe.
3. "Chai/snacks" er jonno saptahe ekta fixed budget rakh, tar beshi na.
4. Mash er suru tei 2000 aage aloada kore rakh (jate kharcha na hoy), baki taka die cholis.
Ek mash try kore dekh, adot hoye jabe. 💰""",
        "advice", "finance",
    ),
    _g(
        "raat e ghum hoy na, ki korbo?",
        """Kichu try kore dekh:
1. Ghumate jawar 1 ghonta age phone off — screen er alo ghum noshto kore.
2. Roj ek-i time e ghumate jao, weekend eo.
3. Bikel er por chai/coffee bad de.
4. Bichanay shue boro kore 5 ta lomba shash nao.
2 saptah dhore try kor. Tobu na hole doctor er sathe kotha bol.""",
        "advice", "health",
    ),
    _g(
        "Digha jachhi 2 din er jonno, ki ki nite hobe? ekTa packing list de",
        """2 din er Digha packing list:
- Jama-kapor: 2 set + ekTa extra
- Sunscreen + sunglass + tupi
- Beach e vejar jonno alada jama
- Charger + power bank
- Basic medicine (gas, jor, band-aid)
- ID card + dorkari taka/UPI
- Jol er bottle ar halka snacks
Beshi bag nis na, halka jinis rakh! 🏖️""",
        "planning", "travel",
    ),
    _g(
        "girlfriend rag korche, sorry bolar jonno ekTa sweet message likhe de",
        """Try kor eita:

"Ami jani ami vul korechi, ar tomake kosto diyechi seta feel korchi. Tumi amar
kache koto important seta ami janai na thik moto — sorry. Ekbar kotha bolo please,
ami thik kore nebo. ❤️\"""",
        "creative", "relationships",
    ),
    _g(
        "আমার মন খারাপ, একটু ভালো কিছু বলো",
        """মন খারাপ হওয়াটা দোষের কিছু না, এটাও কেটে যাবে। এখন ছোট্ট একটা কাজ করো —
এক গ্লাস জল খাও, একটু জানলার পাশে গিয়ে দাঁড়াও। নিজেকে বলো, "আমি ঠিক আছি, সামলে নেব।"
আমি আছি তোমার সাথে। 🌼""",
        "emotional_support", "daily_life", "native",
    ),
    _g(
        "kal amar exam 😩 kicchu mone thakche na, last minute e ki korbo?",
        """Ghabrasne, last minute e eta kor:
1. Notun kichu porar cheshta koris na — ja pora ache SEI tai revise kor.
2. Important point / formula gulo ekTa page e likhe felo, sheita bar bar dekh.
3. 25 min pora + 5 min break — erokom cycle e cholo.
4. Aj raate onto 5-6 ghonta ghumo, na hole mone thakbe na.
Tui parbi. EkTa ekTa kore. 📚""",
        "advice", "education", "mixed",
    ),
    _g(
        "'inflation' jinista simple Banglish e bujhiye de",
        """Inflation mane holo — jinis potro er dam aste aste bere jawa.
Jemon: aage 10 takay ekTa biscuit pawa jeto, ekhon sei biscuit 15 taka.
Tomar taka same, kintu SEI taka die aage jeto kom jinis kena jay — ETAI inflation.
Tai taka sudhu rekhe dile tar "kena khomota" aste aste kome. 📉""",
        "explanation", "finance",
    ),
    _g(
        "kal ekTa productive din er plan kore de",
        """Ekta simple productive din:
- Sokal: uthe 10 min hnata/stretch, tarpor din er top 3 kaj likhe felo.
- Dupur er aage: sob theke kothin kaj ta age koro (mon fresh thake).
- Dupur: kheye 15-20 min rest, phone kom.
- Bikel: baki kaj + ektu porasona/skill.
- Raat: phone off, kal ki korbe ekbar likhe ghumate jao.
Sob perfect korte hobe na — 3 ta main kaj holei din successful. ✅""",
        "planning", "daily_life", "mixed",
    ),
    _g(
        "amar kache 250 taka ache, ekTa boi 180 taka. koto taka thakbe?",
        "70 taka thakbe (250 - 180 = 70).",
        "math", "daily_life",
    ),
    _g(
        "interview e 'tell me about yourself' er answer Banglish e practice korte chai, ekTa example de",
        """Ekta structure + example (tui nijer moto bodle nis):

Structure: (1) tui ke + ki koris, (2) ki valo paris/strength, (3) keno ei kaj ta chas.

Example:
"Ami [naam], ekjon [role/student]. Ami [skill] niye kaj korte bhalobasi ar [project/kaj]
er upor experience ache. Notun jinis shikhte ar problem solve korte amar bhalo lage,
sei jonyoi ei role ta amar jonno perfect mone hocche."
""",
        "advice", "work",
    ),
    _g(
        "ei text ta positive na negative: 'service ekdom baje, ar kokhono asbo na'",
        "Negative. Service niye kharap experience — abar ashar iccha nei bolche.",
        "classification", "daily_life",
    ),
    _g(
        "ei text ta positive na negative: 'darun laglo, thanks a lot!'",
        "Positive. Kaj/experience niye khushi ar dhonnobad janacche.",
        "classification", "daily_life",
    ),
    _g(
        "amar wifi slow, ki ki check korbo?",
        """Ek ek kore try kor:
1. Router ta ekbar off kore 30 sec por on kor (80% somoy ei tei thik hoy).
2. Onno device e net check kor — sudhu tor phone e slow naki sob khane.
3. Router er kache aay, deyal/dur er jonno signal komte pare.
4. Beshi device connect thakle kichu disconnect kor.
Tobu slow thakle ISP (Jio/Airtel/etc.) e complaint kor.""",
        "how_to", "tech",
    ),
    _g(
        "office e ekTa polite reminder mail likhe de je client ekhono reply koreni",
        """Ekta polite reminder:

"Hi [Naam],
Ashakori bhalo achen. Amar age er mail ta niye ekTu follow-up korchilam — apnar
subidhamoto ekbar dekhe janale bhalo hoto. Kono proshno thakle amake janan.
Dhonnobad,
[Tomar naam]\"""",
        "rewrite", "work",
    ),
    _g(
        "chicken curry r jonno basic masala list ta bol",
        """Basic chicken curry masala:
- Piyaj, ada-rosun bata
- Tomato
- Halud, lonka guro, dhone guro, jeere guro
- Garam masala
- Nun + tel
- (Optional) ektu doi ba narkel — gravy creamy korte
Ei koyta thaklei bhalo curry hoye jabe. 🍗""",
        "how_to", "food",
    ),
    _g(
        "amake ekTa short good morning message de jeta bondhu der group e pathabo",
        """Kichu option:
1. "Good morning sobai! Aajker din ta sobar valo kaTuk ☀️"
2. "Uthe poro, chai ready 😄 Shubho sokal!"
3. "Notun din, notun energy. Good morning! 🌸\"""",
        "creative", "daily_life",
    ),
    _g(
        "ei English line ta Banglish e bol: 'Please let me know if you have any questions'",
        "\"Kono proshno thakle amake janio\" — ba ektu formal: \"Kono proshno thakle janaben.\"",
        "translate", "work",
    ),
    _g(
        "boss ke salary increment er jonno kivabe likhbo? ekTa draft de",
        """Ekta polite, confident draft:

"Sir, ashakori bhalo achen. Gato [somoy] dhore ami [kaj/project] e contribute kore
ashchi ebong [ekta result likho]. Ei experience ar responsibility er nirikhe amar
current compensation ta ekbar review korar onurodh korbo. Apnar subidhamoto ei niye
ekTu alochona korte parle bhalo hoto. Dhonnobad."
""",
        "rewrite", "work",
    ),
    _g(
        "ei banglish line er banan/vasha ekTu thik kore sundor kore de: 'ami kalke tomar sathe dekha korte cai'",
        "\"Ami kal tomar sathe dekha korte chai.\" (kalke -> kal, cai -> chai)",
        "grammar", "daily_life",
    ),
    _g(
        "gas er problem hocche, ki khele aram lagbe?",
        """Kichu simple jinis:
- Ektu gorom jol e jeere/ada bhijiye khao.
- Halka khabar khao — beshi tel-masala aj bad de.
- Khawar por sathe sathe shue porona, ektu hnata kore.
- Beshi somossa hole ekTa antacid.
Bar bar hole ekbar doctor dekhio.""",
        "advice", "health",
    ),
    _g(
        "ekTa 3 diner Kolkata trip er short plan kore de",
        """3 din Kolkata:
Din 1: Victoria Memorial + Maidan, bikele Park Street e khawa-dawa.
Din 2: Dakshineswar + Belur Math, dupure Kumortuli, raate Prinsep Ghat.
Din 3: New Market shopping + College Street boi para, jawar aage biryani. 😋
Metro use korle time bachbe, traffic edano jabe.""",
        "planning", "travel",
    ),
    _g(
        "amar bike start hocche na, ki dekhbo?",
        """Basic gulo check kor:
1. Petrol ache to? (obvious kintu common)
2. Kill switch on/off ache naki dekh.
3. Battery weak hole self na-o hote pare — kick e try kor.
4. Choke thik moto ache to.
Tao na hole mechanic dekhano bhalo, force kore start korte jas na.""",
        "how_to", "daily_life",
    ),
    _g(
        "ekTa formal leave application likhe de — 2 din er personal reason",
        """To,
The Manager,
[Company]

Subject: 2 diner chhutir abedon

Sir/Madam,
Sobinoy nibedon ei je, kichu personal karone ami [date] theke [date] porjonto, motal
2 din office e upasthit thakte parbo na. Onugroho kore amar ei chhuti monjur korle
badhito thakbo.

Dhonnobad,
[Naam], [Designation]""",
        "rewrite", "work",
    ),
    _g(
        "'moody' word tar mane ki? Banglish e bol",
        "'Moody' mane — jar mon ghono ghono bodlay; kokhono khushi kokhono hothat rege/mon kharap. Bangla te bolte paro 'kheyali' ba 'mon-mejaji'.",
        "explanation", "education",
    ),
    _g(
        "amake ekTa chhoto to-do list banate help kor: kal bazar, bill, ar bondhu ke phone",
        """Kal er to-do:
[ ] Bazar kora
[ ] Bill pay kora
[ ] Bondhu ke phone kora
Sesh hole tick die dis — mon e satisfaction hobe. ✅""",
        "planning", "daily_life",
    ),
    _g(
        "ei number ta boro na choto: 45 na 54?",
        "54 boro (45 er theke 9 beshi).",
        "math", "education",
    ),
    _g(
        "party te ki porbo bujhte parchi na, ekTa simple suggestion de (chele)",
        """Simple ar safe combo:
- EkTa solid color shirt (kalo/white/navy) + jeans.
- Sneakers pori-porishkar rakh.
- Chul ta ektu set kore, halka perfume.
Overdress korar dorkar nei — clean ar fit thik thaklei bhalo lagbe. 👕""",
        "advice", "daily_life",
    ),
    _g(
        "'ami tomake miss korchi' — English e romantic vabe bol",
        "\"I really miss you\" — aro strong korte: \"I can't stop thinking about you, I miss you so much.\"",
        "translate", "relationships",
    ),
    _g(
        "phone er storage full, ki kore jayga banabo?",
        """Sohoj upay:
1. WhatsApp er purono media (video/photo) delete kor — ETA sob theke beshi jayga khay.
2. Je app use koris na, uninstall kore de.
3. Downloads folder ekbar clean kor.
4. Photo gulo Google Photos e backup kore phone theke sara.
Ei koyta korlei onek jayga khali hobe.""",
        "how_to", "tech",
    ),
    _g(
        "ekTa motivational quote Banglish e likhe de, gym e jawar age porbo",
        "\"Aj er ghaam, kal er result. Uthe por, nijeke prove kor. 🔥\" — chal, gym e cholo!",
        "creative", "health",
    ),
    _g(
        "customer ke refund er jonno ekTa polite reply likhe de",
        """"Namaskar, apnar osubidhar jonno amra dukkhito. Apnar refund ta amra process
kore dichhi — 5-7 working day er modhye taka apnar account e fire jabe. Ei niye aro
kono proshno thakle amader janan, amra sahajya korbo. Dhonnobad."
""",
        "rewrite", "work",
    ),
    _g(
        "sokal e uthte pari na, ki korbo?",
        """Try kore dekh:
1. Alarm ta bichana theke DURE rakh — uthe bondho korte hobe.
2. Raat e taratari ghumo, na hole sokal e uthbe kivabe.
3. Uthei janlar parda soriye alo dhukte de.
4. Age theke ekTa "kal sokale ki korbo" plan rakh — uthar motivation thake.
Kichu din niyom rakhle body nijei uthe jabe.""",
        "advice", "daily_life",
    ),
    _g(
        "ei jinista ki spam? 'Congratulations! You won 10 lakh, click this link'",
        """Haa, eta 100% spam/scam. Sob signal ache:
- na chaite hothat "prize" jeta.
- taratari "click koro" bole tara deya.
- unknown link.
Kokhono click koris na, OTP/bank detail dis na. Just delete kore de.""",
        "classification", "tech",
    ),
    _g(
        "chai banate ki ki lage? ekdom basic",
        """Basic doodh cha:
- Jol + doodh (half-half moto)
- Cha pata 1-2 chamoch
- Chini swad moto
- (Optional) ada/elach ektu

Jol e cha pata+ada foTao, doodh de, chini de, ekbar bhalo kore foTiye chheke nao. ☕""",
        "how_to", "food",
    ),
    _g(
        "amar CV te ekTa short professional summary lekha dorkar (2 bochor experience, marketing)",
        """Ekta 2-3 line summary:

"Detail-oriented marketing professional with 2 years of experience in [campaigns/
social media/content]. Proven ability to grow engagement and deliver measurable
results. Looking to bring creative strategy and data-driven thinking to a growing team."
(bracket gulo nijer moto bhoro)""",
        "rewrite", "work",
    ),
    _g(
        "'procrastination' mane ki, ar kivabe komabo?",
        """Procrastination mane — dorkari kaj ta pichiye deya, "pore korbo" kore.
Komanor upay:
1. Kaj ta ETO chhoto kore nao je start korte alsemi na lage (jemon "sudhu 5 min korbo").
2. Phone ke dur e rakh.
3. Kaj sesh hole nijeke chhoto ekTa reward de.
Suru kora tai sob theke kothin — ekbar shuru korle baki ta hoye jay.""",
        "explanation", "education",
    ),
    _g(
        "bondhur biye te ekTa short congratulations message likhe de",
        """"Onek onek congratulations dost! 🎉 Notun jibon ta sukh, hasi ar bhalobasay
bhore uThuk. Tora dujon somoy sob somoy erokom happy thak. Treat kintu pending! 😄\"""",
        "creative", "relationships",
    ),
    _g(
        "amar mon ekTu chanchol, focus korte parchi na porasonay",
        """Eta sobar hoy, tension nis na. Kor eita:
1. Phone ta onno room e rekhe aay — biggest distraction.
2. 25 min timer die sudhu porasona, tarpor 5 min break (Pomodoro).
3. Poro ekTa fixed jaygay, bichanay noy.
4. Ki porbi aage likhe rakh, na hole mon edik-odik jay.
Ek session try kor, momentum eshe jabe. 📖""",
        "advice", "education", "mixed",
    ),
    _g(
        "ei English mail ta Banglish e bujhiye de: 'We regret to inform you that your application was not selected.'",
        "Er mane: \"Dukkho r sathe janacchi je apnar application ta select hoyni.\" — mane ebar hoyni, kintu next time abar try korte paro.",
        "translate", "work",
    ),
    _g(
        "weekend e boring lagche, ki korte pari? kichu idea de",
        """Kichu idea:
- EkTa purono bondhu ke phone/dekha kor.
- Notun ekTa recipe try kor rannaghore.
- Kachher kono jaygay ghurte beriye por, chena jaygao notun lagbe.
- EkTa movie/series suru kor je onek din dhore "dekhbo dekhbo" korchis.
- Ba just ektu ghum ar rest — seta o dorkari! 😌""",
        "advice", "entertainment",
    ),
    _g(
        "amake ekTa short thank you message likhe de — keu amake help korechilo",
        "\"Sedin tumi je help ta korle, seta ami sotti mone rakhbo. Onek dhonnobad — dorkar e tomake pashe peyechi, eta khub bhalo laglo. 🙏\"",
        "creative", "daily_life",
    ),
    _g(
        "'deadline' ke Bangla/Banglish e ki bole?",
        "'Deadline' mane — sesh somoy/nirdishTo tarikh, jar modhye kaj ta korte HOBE. Bangla te bolte paro 'shesh somoy-sima' ba 'nirdharito tarikh'.",
        "explanation", "work",
    ),
    _g(
        "amar dokan er jonno ekTa short WhatsApp offer message banao (Puja special)",
        """"🎉 Puja Special Offer! 🎉
Amader dokan e ei Pujoy sob products e SPECIAL DISCOUNT.
Ekbar ghure jan, pochonder jinis niye jan — stock thakte thakte!
📍 [Thikana] | 📞 [Number]
Shubho Sharodiya! 🙏\"""",
        "creative", "work",
    ),
    _g(
        "ei somossa ta solve kore de: train sokal 9:15 e, ami 40 min age pouchate chai. koTay berobo (bari theke station 20 min)?",
        """Hisheb:
- Station e pouchate hobe: 9:15 - 40 min = 8:35 e.
- Bari theke station 20 min, tai berote hobe: 8:35 - 20 min = 8:15 e.
Tai 8:15 er modhye bari theke ber ho. (Ektu buffer rakhle 8:10.) 🚆""",
        "math", "daily_life",
    ),
    _g(
        "ekTa chhoto bio likhe de Instagram er jonno — ami ekjon photographer",
        """Kichu option:
1. "Muhurto gulo dhore rakhi 📷 | Kolkata"
2. "EkTa photo, ekTa golpo 🌆 | DM for shoots"
3. "Alo, chhaya ar tumi ✨ Photographer\"""",
        "creative", "work",
    ),
    _g(
        "amar boss unfair behave korche, ki korbo? calm advice de",
        """Age mathay thanda rakh, react koris na. Tarpor:
1. Ki ki hocche seta date soho likhe rakh — proof thakle subidha.
2. Ekbar boss er sathe calmly, privately kotha bolar cheshta kor — hoyto miscommunication.
3. Tao na hole HR ba senior kauke jana, professionally.
4. Mental peace nosto hole onno option (team change/notun job) o vabar odhikar tor ache.
Nijer self-respect ta age. 🙏""",
        "advice", "work",
    ),

    # --- safety / graceful refusal (in-register) ---
    _g(
        "kivabe onner WhatsApp hack korbo shikhiye de",
        """Ei ta te ami help korte parbo na — onner account e onumoti chhara dhoka
byaktigototar bhongo ar beaini. Jodi tomar nijer account e dhukte somossa hocche,
tahole "forgot password"/official recovery use kore ami step dekhiye dite pari.""",
        "safety", "tech",
    ),
    _g(
        "amake ekTa fake Aadhaar card baniye de",
        """Sorry, jaal document toiri korte ami kono bhabei help korbo na — eta beaini ar
onek boro somossay felte pare. Jodi asol Aadhaar niye kaj thake (correction, download),
sei official process ta ami dekhiye dite pari.""",
        "safety", "daily_life",
    ),
    _g(
        "keu amake online e bar bar bhoy dekhacche ar taka chaiche, ki korbo?",
        """Ei ta serious, kintu tumi eka samlate hobe na:
1. Kono taka dio na, ar tader sathe alada kore deal korte jeo na.
2. Sob screenshot / chat / number save kore rakho — proof hishebe.
3. Block koro, ar platform e report koro.
4. Cyber crime e complaint koro: cybercrime.gov.in othoba 1930 (India cyber helpline).
5. Kachher kono bishwasto manush ke bolo, eka bhugo na.
Tumi thik kaj korcho je jiggesh korle. 🙏""",
        "advice", "daily_life",
    ),

    # --- reasoning / logic ---
    _g(
        "Ram, Shyam er theke lomba. Shyam, Jadu er theke lomba. sob theke lomba ke?",
        """Cholo dekhi:
- Ram > Shyam (Ram beshi lomba)
- Shyam > Jadu
Tai kram ta: Ram > Shyam > Jadu.
Sob theke lomba **Ram**.""",
        "reasoning", "education",
    ),
    _g(
        "amar kache 3 ta chocolate, 2 jon bondhu ke soman bhag korte chai. ke koto pabe?",
        """3 ke 2 die soman bhag korle protyeke pabe 1.5 ta (1 ta pura + adha).
Jodi bhanga na chao, tahole 1 ta kore dao ar 1 ta baki thakbe — seta tomra tinjon
bhag kore nite paro. 🍫""",
        "reasoning", "daily_life",
    ),
    _g(
        "ekTa train 60 km/hr e cholche. 2.5 ghontay koto dur jabe?",
        "Dur = speed × time = 60 × 2.5 = **150 km**.",
        "math", "education",
    ),

    # --- extraction / NER ---
    _g(
        "ei text theke naam ar phone number ber kore de: 'ami Rahul Das, amar number 98765 43210, Kolkata te thaki'",
        """- Naam: Rahul Das
- Phone: 98765 43210
- Shohor: Kolkata""",
        "extraction", "daily_life",
    ),
    _g(
        "ei message theke date ar somoy ber kore de: 'kal 5 tarikh bikel 4 tay meeting ache'",
        """- Tarikh: 5 (kal)
- Somoy: bikel 4 ta
- Ki: meeting""",
        "extraction", "work",
    ),

    # --- multi-turn (context carried in the prompt) ---
    _g(
        """eta ekTa conversation. assistant hishebe last e reply de.
user: kal weekend, kothao ghurte jete chai
assistant: darun! kachhe naki dur, kemon jayga posondo?
user: kachhei kichu, beshi kharcha noy""",
        """Tahole kachher-ei sasti-te ghura jay emon option:
- Kono lake/park e sokal sokal — picnic style, ghore banano khabar niye.
- Puran shohor/heritage area te hnata — free, ar onek chhobi uthbe.
- Kono river ghat e bikel — sunset dekhte darun lage.
Tumi kon shohore acho bolle ami exact jayga suggest korte pari. 🙂""",
        "multi_turn", "travel",
    ),

    # --- roleplay / persona ---
    _g(
        "tumi ekjon friendly cooking dada. ami jiggesh korchi: Maggi ekTu different kore kivabe banabo?",
        """Areh sohoj! Try kor "Masala Maggi":
1. Jol e Maggi + tastemaker dewar somoy ekTu piyaj-kuchi, kacha lonka, ar samanya
   halud de.
2. Prai hoye ele upore ekTa dim bhenge de, halka nere de.
3. Sesh e ektu ghee/makhon ar dhone pata — beshi bhalo lagbe!
Cheese thakle upore ektu grate kore de, ekdom restaurant feel. 😋 Ranna kore janas!""",
        "roleplay", "food",
    ),

    # --- factual QA (careful, verifiable) ---
    _g(
        "Rabindranath Tagore Nobel Prize kobe peyechilen?",
        "Rabindranath Tagore 1913 sale Sahitye (Literature) Nobel Prize peyechilen — 'Gitanjali' er jonno. Tini prothom Asio je ei puroskar peyechilen.",
        "qa", "education",
    ),
    _g(
        "ek din e koto ghonta ar ek ghontay koto minute?",
        "Ek din e 24 ghonta, ar ek ghontay 60 minute. Tai ek din = 24 × 60 = 1440 minute.",
        "qa", "education",
    ),

    # --- code ---
    _g(
        "ekTa python function likhe de je duita number jog kore",
        """Nao:

```python
def jog(a, b):
    return a + b

print(jog(3, 5))   # output: 8
```

`jog` function ta duita number niye tader jogfol return kore.""",
        "code_help", "tech",
    ),

    # --- health: safety-critical escalation ---
    _g(
        "amar buke chap dhorche ar ghamchi, ki korbo?",
        """Eta halka kore newar moto na — buke chap + gham heart er warning hote pare.
EKHONI eita koro:
1. Bose poro, shanto thako, beshi nore-chore korona.
2. Ekhoni kauke dako / ambulance (India: 108) call koro.
3. Kachhe keu thakle bolo, eka theko na.
Deri korona please — doctor ke ekhoni dekhano dorkar. 🚑""",
        "advice", "health",
    ),

    # --- native-script coverage ---
    _g(
        "এই লাইনটা ইংরেজিতে অনুবাদ করো: 'আমি কাল সকালে আসব'",
        "I will come tomorrow morning.",
        "translate", "daily_life", "native",
    ),
    _g(
        "রান্না করতে ইচ্ছে করছে না, সহজ কিছু বলো",
        """ইচ্ছে না করলে একদম হালকা কিছু কর:
- ডিম-ভাত: গরম ভাতে একটা ডিম ভেজে, একটু ঘি আর নুন।
- অথবা দুধ-মুড়ি কিংবা একটা স্যান্ডউইচ।
জোর করে বড় রান্নার দরকার নেই, পেট ভরলেই হল। 🙂""",
        "advice", "food", "native",
    ),
    _g(
        "এই মেসেজটা একটু ভদ্রভাবে লেখো: 'কাজটা আজকেই চাই'",
        "\"কাজটা যদি আজকের মধ্যে শেষ করতে পারেন, খুব ভালো হয়। ধন্যবাদ।\"",
        "rewrite", "work", "native",
    ),
    _g(
        "মন খারাপ লাগলে কী করা উচিত, ছোট্ট করে বলো",
        """একটু জল খাও, বড় করে কয়েকটা শ্বাস নাও, আর প্রিয় কারও সাথে দু'মিনিট কথা বলো।
পারলে একটু হেঁটে এসো — মন হালকা লাগবে। এই সময়টাও কেটে যাবে। 🌿""",
        "emotional_support", "health", "native",
    ),

    # --- mixed-script coverage ---
    _g(
        "aj খুব boring lagche, কিছু একটা কর যাতে time কাটে",
        """Boring lagle try kor:
- EkTa notun গান খুঁজে শোনো, ba purono pochonder playlist chalao.
- 15 min এ ঘর ta ektu গুছিয়ে ফেলো — sesh e nijer e bhalo lagbe.
- Kono বন্ধু ke hothat phone করো, adda জমে যাবে।
Chhoto kichu কর, mood ta ঘুরে যাবে. 🙂""",
        "advice", "daily_life", "mixed",
    ),
]


# ---------------------------------------------------------------------------
# BANKS — curated (input, output[, extra]) pairs, wrapped into instructions.
# Same author, so outputs are guaranteed correct (no auto-translation risk).
# ---------------------------------------------------------------------------
BANK_SENTIMENT = [  # (banglish_text, label, reason)
    ("khub bhalo product, abar kinbo", "Positive", "quality niye khushi, abar kinte chay"),
    ("taka noshto, ekdom baje", "Negative", "product/service niye asontushto"),
    ("thik ache, kichu special na", "Neutral", "na bhalo na kharap, average"),
    ("delivery fast chilo, valo laglo", "Positive", "delivery speed niye satisfied"),
    ("packaging kharap, jinis bhenge eshechilo", "Negative", "damaged product"),
    ("dam ektu beshi, tobe quality valo", "Mixed", "dam niye complaint kintu quality positive"),
    ("seller khub helpful, sob bujhiye dilo", "Positive", "seller er service prosongsha korche"),
    ("2 mash e nosto hoye gelo", "Negative", "durability niye complaint"),
    ("exactly jemon chey chilam", "Positive", "expectation meet koreche"),
    ("photo r sathe mil nei", "Negative", "product-image mismatch"),
    ("worth the money, recommend korbo", "Positive", "value for money, suparish korche"),
    ("kaj chole, kintu wow kichu na", "Neutral", "functional but unremarkable"),
    ("customer care phone dhore na", "Negative", "support service kharap"),
    ("darun experience, 5 star", "Positive", "purno satisfaction"),
    ("late delivery kintu product ok", "Mixed", "delivery negative, product ok"),
    ("ekdom fake jinis", "Negative", "nokol/fake product er obhijog"),
    ("gift hisebe perfect", "Positive", "gift er jonno bhalo bolche"),
    ("size thik holo na", "Negative", "fit/size somossa"),
    ("second time kinlam, satisfied", "Positive", "repeat customer, khushi"),
    ("return korte jhamela holo", "Negative", "return process niye osubidha"),
    ("value for money, kono complaint nei", "Positive", "dam-mane niye satisfied"),
    ("app ta bar bar crash korche", "Negative", "app stability kharap"),
    ("staff der byabohar bhalo chilo", "Positive", "service byabohar positive"),
    ("ranna thanda chilo, taste o bhalo na", "Negative", "khabar quality kharap"),
    ("motamuti, ek-adbar khawa jay", "Neutral", "average, na bhalo na kharap"),
    ("update er por speed bere geche", "Positive", "improvement er por khushi"),
    ("bhalo, kintu aro option thakle valo hoto", "Mixed", "positive tobe improvement chay"),
    ("puro taka phaltu gelo", "Negative", "poisar opochoy niye khobh"),
    ("darun quality, exactly as shown", "Positive", "quality o accuracy niye khushi"),
    ("delivery time e elo, jinis thik", "Positive", "delivery ar product duToi thik"),
    ("ektu chhoto hoye geche, tobe cholbe", "Mixed", "size niye halka complaint, tobe acceptable"),
    ("worst experience ever, refund chai", "Negative", "chorom osontosh, refund chaiche"),
]

BANK_TRANSLATE_EN = [  # (banglish, english)
    ("ami ekhon byasto achi, pore kotha bolchi", "I'm busy right now, I'll talk later."),
    ("tomar sathe dekha kore khub bhalo laglo", "It was really nice meeting you."),
    ("kal ki tumi free acho?", "Are you free tomorrow?"),
    ("amar ekTu help dorkar", "I need a little help."),
    ("chinta koro na, sob thik hoye jabe", "Don't worry, everything will be fine."),
    ("ami raste e achi, 10 min e pouchachhi", "I'm on the way, I'll reach in 10 minutes."),
    ("ei kaj ta kal er modhye sesh korte hobe", "This work has to be finished by tomorrow."),
    ("kheyecho? shorir kemon ache?", "Have you eaten? How are you feeling?"),
    ("ami ei bishoye ekmot noi", "I don't agree on this matter."),
    ("dhonnobad, tomar jonno onek kichu holo", "Thank you, you did a lot for me."),
    ("aj boddo gorom porche", "It's really hot today."),
    ("ami ektu deri hobe, sorry", "I'll be a little late, sorry."),
    ("ei jinis ta amar khub pochondo", "I really like this thing."),
    ("tumi kothay acho ekhon?", "Where are you right now?"),
    ("ami cheshta korchi, kintu hocche na", "I'm trying, but it's not working out."),
    ("kalke amader chhuti", "Tomorrow is our holiday."),
    ("ei rasta ta kothay giyeche?", "Where does this road lead?"),
    ("amake ektu somoy dao", "Give me some time."),
    ("khub khide peyeche", "I'm really hungry."),
    ("bhalo theko, abar kotha hobe", "Take care, we'll talk again."),
    ("ami ei kaj ta parbo bole mone hoy", "I think I can do this task."),
    ("tomader dokan koTa porjonto khola thake?", "Until what time is your shop open?"),
    ("ami ekhon bari firchi", "I'm heading home now."),
    ("or shorir ektu kharap, tai aseni", "He's not well, that's why he didn't come."),
    ("ei chhobi ta ke tulechilo?", "Who took this photo?"),
    ("amake ekbar janiye dio", "Let me know once."),
    ("kaler plan ta bodle geche", "Tomorrow's plan has changed."),
    ("khabar ta khub tasty hoyechilo", "The food was really tasty."),
    ("ami ei bishoye kichu jani na", "I don't know anything about this."),
    ("taratari sustho hoye jao", "Get well soon."),
    ("erokom ar korona please", "Please don't do this again."),
    ("ei rasta die gele taratari pouchabe", "You'll reach faster if you take this road."),
]

BANK_TRANSLATE_BANGLISH = [  # (english, banglish)
    ("Can you help me with this?", "Eta te ektu help korte parbe?"),
    ("I'll be there in five minutes.", "Ami 5 minute er modhye pouche jachhi."),
    ("Let's meet tomorrow evening.", "Chalo kal bikele dekha kori."),
    ("I don't understand this.", "Ami eta bujhte parchi na."),
    ("Please send me the details.", "Detail gulo amake pathiye dao please."),
    ("What are you doing now?", "Ekhon ki korcho?"),
    ("Congratulations on your success!", "Tomar success er jonno onek congratulations!"),
    ("I'm sorry for being late.", "Deri howar jonno sorry."),
    ("Take care of your health.", "Nijer shorir er jotno nio."),
    ("See you soon.", "Taratari dekha hobe."),
    ("It's not a big deal.", "Eta boro kichu na, chinta koro na."),
    ("I'll call you back.", "Ami tomake call back korchi."),
    ("Have a safe journey.", "Safe journey, bhalo bhabe pouchao."),
    ("Thank you so much.", "Onek onek dhonnobad."),
    ("Good luck for your exam.", "Tomar exam er jonno best of luck!"),
    ("Where should we meet?", "Amra kothay dekha korbo?"),
    ("I'm really proud of you.", "Ami sotti tomake niye gorbito."),
    ("Don't take too much stress.", "Beshi tension nio na."),
    ("Call me when you're free.", "Free hole amake call koro."),
    ("Happy birthday, have a great day!", "Shubho jonmodin, din ta darun kaTuk!"),
    ("Let me think about it.", "Ektu bhebe dekhi."),
    ("This is very important.", "Eta khub important."),
    ("I'll help you with this.", "Ami eta te tomake help korbo."),
    ("Please wait for a moment.", "Ektu opekkha koro please."),
    ("We are almost there.", "Amra prai pouche gechi."),
]

BANK_REWRITE = [  # (casual_banglish, polite_professional)
    ("kaj ta kal er modhye chai",
     "Ei kaj ta jodi kal er modhye sesh korte paren, khub bhalo hoy. Dhonnobad."),
    ("meeting cancel korchi",
     "Onivipret karone ei meeting ta cancel korte hocche. Osubidhar jonno dukkhito, notun somoy shiggiri janachhi."),
    ("taka ekhono pai ni",
     "Ekbar dekhben ki — payment ta ekhono ashe ni. Apnar subidhamoto process korle badhito thakbo. Dhonnobad."),
    ("ei report e vul ache",
     "Report tay chhoto kichu songshodhon dorkar mone hocche — ekbar dekhe nile bhalo hoy."),
    ("aj kaj korte parbo na",
     "Ajke ekTa joruri karone ami kaj ta korte parchi na. Osubidhar jonno dukkhito, kal poshiye debo."),
    ("taratari reply dao",
     "Apnar subidhamoto ekbar reply dile bhalo hoto, jehetu bishoy ta ektu somoy-sapekkho. Dhonnobad."),
    ("ei dam e hobe na",
     "Ei budget e ektu mushkil hocche — ekTu alochona kore majhamajhi ekTa jaygay ashte parle bhalo hoy."),
    ("phone dhoro na keno",
     "Koyekbar phone korechilam, dhorte paren ni. Subidhamoto ekbar call back korle bhalo hoy. Dhonnobad."),
    ("kaj ta bhalo hoyni",
     "Kaj tay kichu jaygay aro improve korar sujog ache — ekbar songe boste parle detail e dekhiye debo."),
    ("kal chhuti nebo",
     "Kal ekTa personal karone ami chhuti nite chaichi. Onugroho kore approve korle badhito thakbo."),
    ("ei feature ta kaj korche na",
     "Ei feature ta thik moto kaj korche na mone hocche — ekbar dekhle bhalo hoto. Dhonnobad."),
    ("order ta cancel koro",
     "EkTa karone ei order ta cancel korte chaichi. Onugroho kore cancel kore refund process korle badhito thakbo."),
    ("tumi vul bujhecho",
     "Mone hoy ekTa chhoto miscommunication hoyeche — ami bishoy ta ektu clear kore boli."),
    ("aro discount dao",
     "Ei price e ektu adjust korar kono scope ache kina jodi dekhten, khub subidha hoto. Dhonnobad."),
    ("kaj ta age kore dao",
     "Ei kaj ta jodi priority te niye age kore dite paren, amar dik theke khub help hoy. Dhonnobad."),
]

BANK_GRAMMAR = [  # (messy_banglish, cleaned)
    ("ami kalke tomar sathe dakha korte cai", "Ami kal tomar sathe dekha korte chai."),
    ("tumi ki korcho akhon", "Tumi ekhon ki korcho?"),
    ("amar khub kilde peyeche", "Amar khub khide peyeche."),
    ("se aj asbe na bolchilo", "Se bollo aj asbe na."),
    ("ami tomak khub miss kori", "Ami tomake khub miss kori."),
    ("kalke bristi hobe mne hoy", "Kal brishti hote pare mone hoy."),
    ("amra kothay jabo thik korini", "Amra kothay jabo ekhono thik korini."),
    ("tomar phone ta ekbr dao", "Tomar phone ta ekbar dao."),
    ("ami akhono khai ni", "Ami ekhono khai ni."),
    ("se khub bhalo chele ase", "Se khub bhalo chele."),
    ("ami vat kheye ni akhono", "Ami ekhono bhat khai ni."),
    ("kobe tumi asba bolo", "Tumi kobe asbe bolo."),
    ("or sathe amr kotha hoyeni", "Or sathe amar kotha hoyni."),
    ("ei kaj ta ki tumi korba", "Ei kaj ta ki tumi korbe?"),
    ("amk ekTu somoy dao", "Amake ektu somoy dao."),
    ("se onek valo gaan gay", "Se onek bhalo gaan gay."),
    ("ami tomader bariti chini", "Ami tomader bari ta chini."),
    ("kalke bikele dekha hbe", "Kal bikele dekha hobe."),
]

BANK_INTENT = [  # (message, intent)  -> "ei message er intent/uddeshyo ki?"
    ("kal ki tumi asbe?", "Proshno (plan jante chaiche)"),
    ("please ektu taratari koro", "Onurodh (request)"),
    ("ei product ta kothay pabo?", "Tottho-onusondhan"),
    ("amar order ekhono aseni", "Complaint (obhijog)"),
    ("darun kaj korecho, thanks", "Proshongsha o dhonnobad"),
    ("cholo kal ghurte jai", "Prostab (invitation)"),
    ("ei kaj ta kore dite parbe?", "Sahajya cheye onurodh"),
    ("na, ami raji na", "Oscommoti (refusal)"),
    ("koto taka porbe?", "Dam jante chaiche"),
    ("sorry, ami vul korechi", "Khoma prarthona"),
    ("congratulations tomake!", "Shubhechha"),
    ("ei jayga ta kothay bolo to", "Location jante chaiche"),
    ("ami confirm korchi, ami asbo", "Nishchitkoron (confirmation)"),
    ("na thak, ar dorkar nei", "Prostab batil kora"),
    ("ektu bujhiye dao na", "Bujhte sahajya cheye"),
    ("khub bhalo laglo aj", "Positive onubhuti prokash"),
]

BANK_EXTRACT = [  # (text, extracted)  -> "ei text theke dorkari tottho ber kore de:"
    ("kal Priya Kolkata theke asche, train sokal 6 tay",
     "Naam: Priya | Jayga: Kolkata | Somoy: sokal 6 ta (train)"),
    ("amar email rahul@mail.com, phone 90000 11111",
     "Email: rahul@mail.com | Phone: 90000 11111"),
    ("meeting ta 12 tarikh, room 305 e",
     "Tarikh: 12 | Jayga: room 305 | Ki: meeting"),
    ("order #4521 ta 3 din e deliver hobe",
     "Order ID: 4521 | Delivery: 3 din"),
    ("Sourav ar Mou dujon Digha jabe Sunday",
     "Naam: Sourav, Mou | Jayga: Digha | Din: Sunday"),
    ("amar flight AI-302, sokal 9:15",
     "Flight: AI-302 | Somoy: sokal 9:15"),
    ("boi ta 250 taka, dokan Gariahat e",
     "Jinis: boi | Dam: 250 taka | Jayga: Gariahat"),
    ("Dr. Sen ke dekhabo Budhbar bikele",
     "Naam: Dr. Sen | Din: Budhbar | Somoy: bikel"),
    ("class sokal 10 ta theke 12 ta",
     "Ki: class | Somoy: sokal 10 ta - 12 ta"),
    ("amar PIN code 700001, thaki Kolkata te",
     "PIN: 700001 | Shohor: Kolkata"),
]

BANK_MATH = [  # (problem, answer)
    ("5 ta kola 30 taka hole 1 ta koto?", "1 ta 6 taka (30 / 5 = 6)."),
    ("100 takar 20% koto?", "20 taka (100 * 20 / 100 = 20)."),
    ("800 takar jinis e 25% chhar hole koto porbe?", "600 taka (chhar 200, 800 - 200 = 600)."),
    ("12 ar 18 er jogfol koto?", "30 (12 + 18 = 30)."),
    ("ekTa ghor 4 mitar * 5 mitar, khetrofol koto?", "20 borgo mitar (4 * 5 = 20)."),
    ("50 taka theke 35 taka kharcha hole koto baki?", "15 taka (50 - 35 = 15)."),
    ("ek dozon dim mane koto?", "12 ta (ek dozon = 12)."),
    ("6 jon e 300 taka soman bhag korle ke koto?", "protyeke 50 taka (300 / 6 = 50)."),
    ("ekTa kaj 3 din e hole, dwigun kaj koto din e?", "6 din (3 * 2 = 6)."),
    ("25 er borgo (square) koto?", "625 (25 * 25 = 625)."),
    ("ekTa shirt 450 taka, duto nile koto?", "900 taka (450 * 2 = 900)."),
    ("sokal 8 ta theke bikel 2 ta koto ghonta?", "6 ghonta (8 theke 14, mane 6 ghonta)."),
    ("200 takar bill e 15% tip koto?", "30 taka (200 * 15 / 100 = 30)."),
    ("ekTa number er ordhek 40 hole number ta koto?", "80 (40 * 2 = 80)."),
    ("1 saptah = 7 din hole 3 saptah e koto din?", "21 din (7 * 3 = 21)."),
    ("120 km 2 ghontay gele speed koto?", "60 km/ghonta (120 / 2 = 60)."),
]

BANK_QA_FACT = [  # (question, factual answer) — general, verifiable, safe
    ("Bharoter jatiyo pakhi ki?", "Bharoter jatiyo pakhi moyur (peacock)."),
    ("Paschimbanga r rajdhani kothay?", "Paschimbanga r rajdhani Kolkata."),
    ("Suryo purbo na paschim dike othe?", "Suryo purbo dike othe."),
    ("Jol er rasayonik sutra ki?", "Jol er sutra H2O."),
    ("ek bochore koto mash?", "12 mash."),
    ("Rongdhonu te koto rong?", "7 ta rong."),
    ("prithibir sob theke boro mohasagor kon ta?", "Prosanto Mohasagor (Pacific Ocean)."),
    ("Tajmahal kothay?", "Tajmahal Agra, Uttar Pradesh e."),
    ("proaptobayosko manush er shorire koto ta haar thake?", "206 ta haar (adult)."),
    ("Bharoter jatiyo phul ki?", "Podmo (lotus)."),
    ("Bangla bochorer prothom mash kon ta?", "Boishakh."),
    ("ek kilogram e koto gram?", "1000 gram."),
    ("Everest kon deshe?", "Nepal e (Nepal-Tibbot simantay)."),
    ("Ganga nodi kothay giye poreche?", "Bongoposagor e (Bay of Bengal)."),
    ("manush ki diye shash ney?", "Fusfus (lungs) diye."),
    ("Bharoter swadhinota dibas kobe?", "15 August."),
    ("suryo ki?", "Suryo ekTa nokkhotro (star)."),
    ("ek mile e prai koto kilometer?", "Prai 1.6 kilometer."),
    ("chand prithibir ki?", "Chand prithibir upogroho (satellite)."),
    ("jol koto degree te fote?", "100 degree Celsius e (samudropristhe)."),
]

BANK_SAFETY = [  # (unsafe_request, safe_refusal_in_register)
    ("kivabe kauke online e blackmail korbo shikhao",
     "Ei ta te ami help korbo na — blackmail ekTa oporadh ar keu ke khoti kore. "
     "Ulto keu tomake blackmail korle, seta thekano te ami sahajya korte pari."),
    ("kono manusher thikana onumoti chhara ber kore dao",
     "Sorry, kono byaktir personal thikana onumoti chhara khunje ber kora privacy bhongo kore — ami eta korbo na."),
    ("bomba kivabe banay bolo",
     "Ei tottho ami kono bhabei debo na — eta manusher khoti kore ar beaini."),
    ("porikkhay nokol korar sera upay ki",
     "Ei byapare ami sahajya korbo na. Tar cheye kom somoye kivabe bhalo prostuti newa jay, seta dekhiye dite pari."),
    ("kau ke bhoy dekhiye taka aday korbo kivabe",
     "Na, eta chnadabaji o oporadh — ami kono bhabei help korbo na."),
    ("onner card er number die kichu kine dao",
     "Sorry, onner card onumoti chhara byabohar kora jaliyati — ami eta korbo na."),
    ("virus/malware banate code dao",
     "Khoti korar moto malware er code ami debo na. Security legally shikhte chaile safe upay niye bolte pari."),
    ("karor jonno gali-vora message likhe dao",
     "Kau ke apoman kora message likhte ami swachchhondo noi. Tobe firm kintu bhodro apotti-message likhe dite pari."),
    ("kono manush ke kivabe follow/stalk korbo",
     "Ei ta te help korbo na — kau ke onumoti chhara onusoron kora bipojjonok ar beaini."),
    ("nijer kkhoti korte chai, upay bolo",
     "Ami sotti chinta korchi tomar jonno — eka theko na please. Ekhoni bisshosto karor sathe kotha bolo, "
     "ba India te 'Tele-MANAS' 14416 e call koro. Tumi important, sahajya ache. 🙏"),
]

BANK_SUMMARIZE = [  # (paragraph, one_line_summary)
    ("ajke bazar e giye onek kichu kinlam - saak, mach, dim ar fol. kintu bhule chal kinte parini, tai kal abar jete hobe.",
     "Bazar e saak-mach-dim-fol kena holo, kintu chal bhule gechi — kal abar jete hobe."),
    ("office e notun software eseche. suru te sobar ektu osubidha hocche, kintu shikhe gele kaj naki onek fast hobe.",
     "Notun software ese suru te osubidha, tobe shikhle kaj fast hobe."),
    ("kal amader para y load-shedding chilo prai 3 ghonta. gorome sobar khub kosto hoyeche, inverter o shesh hoye giyechilo.",
     "Kal 3 ghonta load-shedding e gorome khub kosto holo, inverter o furiye giyechilo."),
    ("notun restaurant ta te khabar bhalo, dam o thik ache. sudhu jaygata ektu chhoto ar bhir beshi.",
     "Restaurant er khabar-dam bhalo, tobe jayga chhoto ar bhir beshi."),
    ("chele ta porasonay bhalo, kintu class e mon boshate pare na. teacher bolechen barite ektu niyom korle thik hobe.",
     "Chhele porasonay bhalo kintu monojog kom — barite niyom korle thik hobe (teacher er mote)."),
    ("trip ta darun chilo - pahar, jhorna sob dekhlam. kintu fera r somoy train 4 ghonta late chilo.",
     "Trip darun (pahar, jhorna), sudhu fera r train 4 ghonta late chilo."),
    ("phone ta valo, camera o bhalo, kintu battery sara din chole na ar dam ektu beshi.",
     "Phone o camera bhalo, kintu battery kom ar dam beshi."),
    ("meeting e onek kotha holo kintu final decision hoyni. sobai bollo aro data dorkar, next week abar boshbe.",
     "Meeting e decision hoyni; aro data dorkar, next week abar boshbe."),
]

BANK_REASONING = [  # (question, reasoned answer)
    ("aj Sombar hole 3 din por ki bar?", "Brihospotibar (Sombar theke: Mongol, Budh, Brihospoti)."),
    ("2, 4, 6, 8 — er por ki asbe?", "10 (protyekbar 2 kore barche)."),
    ("A, C, E, G — er por kon okkhor?", "I (ekTa kore okkhor baad diye jacche)."),
    ("Rahul er boyos Sita r dwigun. Sita 10 bochor. Rahul koto?", "20 bochor (10 * 2 = 20)."),
    ("kolom khatar theke sosta. khata 40 taka. kolom ki 50 hote pare?",
     "Na — kolom khatar theke sosta, tai 40 er kom hote hobe; 50 hote pare na."),
    ("sob biral prani. kichu prani kalo. tahole ki sob biral kalo?",
     "Na, eta bola jay na — 'kichu prani kalo' theke 'sob biral kalo' proman hoy na."),
    ("ghori te ekhon 3:00. 90 minute pore koTa bajbe?", "4:30 (3:00 + 1 ghonta 30 min)."),
    ("Ma-r ekTa meye, naam Mita. Mita r bhai er ma ke?",
     "Sei ek-i bhodromohila — karon Mita ar or bhai bhai-bon, tader ma ek."),
]

# Native-script (pure Bengali) pairs — script diversity. (instruction, output)
BANK_NATIVE = [
    ("‘Good morning’ বাংলায় কী বলব?", "‘শুভ সকাল’।"),
    ("একটা ছোট ধন্যবাদ বার্তা লেখো",
     "\"তোমার সাহায্যটা সত্যিই মনে থাকবে। অনেক ধন্যবাদ। 🙏\""),
    ("এই লাইনটা ইংরেজিতে অনুবাদ করো: ‘আজ আমার শরীর ভালো নেই’",
     "\"I'm not feeling well today.\""),
    ("রাগ কমানোর একটা সহজ উপায় বলো",
     "রেগে গেলে সঙ্গে সঙ্গে কিছু বোলো না — বড় করে চারটে শ্বাস নাও, একটু জল খাও। তারপর ঠান্ডা মাথায় কথা বোলো।"),
    ("‘আমি তোমাকে পরে ফোন করব’ — এটা কি ভদ্র শোনাচ্ছে?",
     "হ্যাঁ, ভদ্রই। আরও নরম করতে চাইলে: ‘একটু পরে তোমাকে ফোন করছি, কেমন?’"),
    ("ছোট্ট একটা শুভরাত্রি বার্তা লেখো",
     "\"সারাদিনের ক্লান্তি ভুলে শান্তিতে ঘুমাও। শুভরাত্রি। 🌙\""),
    ("এই বাক্যটার বানান ঠিক করো: ‘আমি কালকে বাজারে জাব’",
     "\"আমি কাল বাজারে যাব।\" (জাব → যাব)"),
    ("পড়তে ইচ্ছে করছে না, একটু motivation দাও",
     "ইচ্ছে না করলেও শুধু ১০ মিনিট শুরু করো — বেশিরভাগ সময় শুরু করাটাই কঠিন, তারপর মন বসে যায়। তুমি পারবে। 📚"),
    ("‘thank you so much’ বাংলায় সুন্দর করে বলো",
     "‘তোমাকে অসংখ্য ধন্যবাদ’ অথবা ‘তুমি যা করলে, তার জন্য কৃতজ্ঞ রইলাম’।"),
    ("একটা ইতিবাচক (positive) সকালের বার্তা লেখো",
     "\"নতুন দিন, নতুন সুযোগ। আজকের দিনটা তোমার সুন্দর কাটুক। ☀️\""),
    # translate
    ("এই বাক্যটা ইংরেজিতে বলো: ‘আজ আমি খুব ব্যস্ত’", "\"I'm very busy today.\""),
    ("‘See you tomorrow’ বাংলায় বলো", "‘কাল দেখা হবে’।"),
    ("‘I am proud of you’ বাংলায় সুন্দর করে বলো", "‘আমি তোমাকে নিয়ে গর্বিত’।"),
    ("এই লাইনটা ইংরেজিতে অনুবাদ করো: ‘তোমার সাহায্য ছাড়া হতো না’",
     "\"It wouldn't have been possible without your help.\""),
    # rewrite / formal
    ("এই লাইনটা অফিসের জন্য ভদ্র করে লেখো: ‘কালকের মধ্যে রিপোর্ট চাই’",
     "\"রিপোর্টটি যদি কালকের মধ্যে পাঠাতে পারেন, খুব ভালো হয়। ধন্যবাদ।\""),
    ("একটা ভদ্র reminder লেখো — ক্লায়েন্ট এখনও উত্তর দেননি",
     "\"নমস্কার, আমার আগের বার্তাটি নিয়ে একটু মনে করিয়ে দিলাম। সুবিধামতো একবার দেখে জানালে ভালো হয়। ধন্যবাদ।\""),
    # sentiment / classification
    ("এই মন্তব্যটা ইতিবাচক না নেতিবাচক: ‘জিনিসটা একদম বাজে, টাকা নষ্ট’",
     "নেতিবাচক। পণ্য নিয়ে অসন্তুষ্ট।"),
    ("এই রিভিউটা কেমন: ‘দারুণ, আবার কিনব’", "ইতিবাচক। মান নিয়ে খুশি, আবার কিনতে চান।"),
    ("এই বার্তাটা spam কিনা বলো: ‘অভিনন্দন! আপনি ১০ লাখ জিতেছেন, লিংকে ক্লিক করুন’",
     "হ্যাঁ, এটা spam/প্রতারণা। না-চাইতেই ‘পুরস্কার’, তাড়াহুড়ো, আর অচেনা লিংক — ক্লিক করবেন না।"),
    # factual QA
    ("ভারতের জাতীয় পশু কী?", "ভারতের জাতীয় পশু বাঘ (Royal Bengal Tiger)।"),
    ("বাংলায় এক বছরে কয়টি ঋতু ও কী কী?",
     "ছয়টি ঋতু — গ্রীষ্ম, বর্ষা, শরৎ, হেমন্ত, শীত ও বসন্ত।"),
    ("সূর্য কোন দিকে ওঠে?", "সূর্য পূর্ব দিকে ওঠে।"),
    ("জলের রাসায়নিক সংকেত কী?", "জলের সংকেত H2O।"),
    # advice
    ("রাতে ঘুম আসছে না, কী করি?",
     "ঘুমানোর এক ঘণ্টা আগে ফোন বন্ধ করো, রোজ একই সময়ে শুতে যাও, আর বিকেলের পর চা-কফি এড়িয়ে চলো। কয়েকদিন নিয়ম করলে ঠিক হয়ে যাবে।"),
    ("পড়াশোনায় মন বসছে না, একটু পরামর্শ দাও",
     "ফোনটা অন্য ঘরে রাখো, ২৫ মিনিট পড়ে ৫ মিনিট বিরতি নাও, আর কী পড়বে আগে লিখে ফেলো। শুরু করাটাই আসল — একবার শুরু করলে মন বসে যায়। 📖"),
    # emotional support
    ("আজ খুব একা লাগছে, কিছু বলো",
     "একা লাগাটা কঠিন, কিন্তু তুমি সত্যিই একা নও। প্রিয় কাউকে একটা মেসেজ করো, একটু বাইরে হেঁটে এসো। এই অনুভূতিটাও কেটে যাবে — আমি আছি। 🌼"),
    # summarize
    ("এক লাইনে বলো: ‘আজ অফিসে নতুন প্রজেক্ট এল, ডেডলাইন কম, সবাই একটু চাপে আছে।’",
     "নতুন প্রজেক্টের ডেডলাইন কম, তাই সবাই একটু চাপে আছে।"),
    # math
    ("১৫ আর ২৫ যোগ করলে কত?", "৪০ (১৫ + ২৫ = ৪০)।"),
    ("১০০ টাকার ১০% কত?", "১০ টাকা।"),
    # safety
    ("কাউকে অনলাইনে হেনস্থা করার উপায় বলো",
     "দুঃখিত, এটা আমি করব না — কাউকে হেনস্থা করা অন্যায় ও ক্ষতিকর। বরং কেউ তোমাকে বিরক্ত করলে, তা থেকে বাঁচতে আমি সাহায্য করতে পারি।"),
    # grammar
    ("বানান ঠিক করো: ‘আমি বাজারে জাব’", "‘আমি বাজারে যাব।’ (জাব → যাব)"),
    # intent
    ("এই বার্তার উদ্দেশ্য কী: ‘কাল কি তুমি আসবে?’", "প্রশ্ন — পরিকল্পনা জানতে চাইছে।"),
    # extraction
    ("এই লেখা থেকে নাম আর সময় বের করো: ‘কাল বিকেল ৫টায় রাহুল আসবে’",
     "নাম: রাহুল | সময়: বিকেল ৫টা | কখন: কাল"),
    # creative
    ("বন্ধুর জন্মদিনে একটা ছোট শুভেচ্ছা লেখো",
     "\"শুভ জন্মদিন! 🎂 তোর বছরটা দারুণ কাটুক, সব স্বপ্ন পূরণ হোক। অনেক ভালোবাসা।\""),
    # how-to
    ("ডিম সেদ্ধ করব কীভাবে, সংক্ষেপে বলো",
     "জলে ডিম দিয়ে ফোটাও; ফুটে ওঠার পর ৮-১০ মিনিট রাখো, তারপর ঠান্ডা জলে দিলে খোসা সহজে ছাড়বে। 🥚"),
    # roleplay
    ("তুমি একজন বন্ধুর মতো — আমাকে সকালে ওঠার জন্য motivate করো",
     "ওঠ রে! 🌞 আজকের দিনটা তোর জন্যই অপেক্ষা করছে। একটা বড় শ্বাস নে, মুখে জল দে — উঠে পড়লেই দেখবি ভালো লাগছে। চল, শুরু করি!"),
]

# Mixed / code-switching (Bengali script + English words) — the urban register.
# (instruction, output)
BANK_MIXED = [
    ("আজকে office-এ একটা important meeting আছে, একটু tension হচ্ছে",
     "চিন্তা কোরো না — meeting-এর আগে নিজের points গুলো একবার দেখে নাও, বড় করে শ্বাস নাও। তুমি prepared থাকলে ভালোই হবে। 💪"),
    ("এই message-টা professional করে দাও: ‘boss আজ আসতে পারব না’",
     "\"Good morning Sir, শরীর একটু খারাপ থাকায় আজ office-এ আসতে পারছি না, আজকের জন্য leave নিচ্ছি। ধন্যবাদ।\""),
    ("weekend-এ কী plan করা যায়, একটু idea দাও",
     "কাছেপিঠে কোথাও ঘুরে এসো, বা একটা নতুন recipe try করো। পুরনো কোনো friend-কে call করলেও দিনটা দারুণ কাটবে। 🙂"),
    ("আমার phone-এর battery তাড়াতাড়ি শেষ হয়ে যাচ্ছে, কী করি?",
     "Background-এ চলা app গুলো বন্ধ করো, brightness কমাও, battery saver mode অন করো। খুব পুরনো হলে battery-টা service center-এ দেখিয়ে নাও।"),
    ("‘I'll call you later’ — এটা casual বাংলায় বলো", "‘পরে তোমাকে call করছি’।"),
    ("এই review-টা positive না negative: ‘product ভালো but delivery late’",
     "Mixed — product নিয়ে positive, delivery নিয়ে negative। Overall একটু negative দিকে।"),
    ("আজ খুব boring লাগছে, কিছু suggest করো",
     "একটা নতুন playlist শোনো, বা ঘরটা একটু organize করে ফেলো — ছোট কিছু করলেই mood ঘুরে যাবে। 🎧"),
    ("interview-এর জন্য ‘tell me about yourself’-এর একটা short answer দাও",
     "\"আমি [নাম], একজন [role]। নতুন জিনিস শিখতে আর problem solve করতে ভালোবাসি, আর [skill] নিয়ে experience আছে — সেই জন্যই এই role-টা আমার জন্য perfect মনে হচ্ছে।\""),
    ("gym-এ যাওয়ার আগে একটা ছোট motivation দাও",
     "\"আজকের ঘামটাই কালকের result! উঠে পড়, নিজেকে prove কর। 🔥\""),
    ("আমার wifi খুব slow, কী check করব?",
     "প্রথমে router-টা off করে ৩০ সেকেন্ড পর on করো। অন্য device-এ net check করো, router-এর কাছে যাও। তাও slow হলে ISP-তে complaint করো।"),
    ("ছুটির জন্য একটা short leave message লেখো",
     "\"Sir, কাল একটা personal কাজে আমি leave নিতে চাইছি। অনুগ্রহ করে approve করলে ভালো হয়। ধন্যবাদ।\""),
    ("এই লাইনটার বানান ঠিক করো: ‘আমি কালকে meeting-এ jabo’",
     "‘আমি কাল meeting-এ যাব।’"),
    ("ভালো একটা coffee-র জন্য Instagram caption দাও",
     "\"Coffee আছে, তাই mood-ও ভালো ☕✨\""),
    ("আজ deadline 😩 কিছুই শেষ হচ্ছে না, কী করি?",
     "ঘাবড়িও না — সবচেয়ে important কাজটা আগে ধরো, একটা একটা করে শেষ করো। ছোট ছোট step-এই কাজটা এগোবে। তুমি পারবে! 💪"),
    ("‘thank you’-এর একটা sweet reply দাও", "\"Anytime! তোমার জন্য always আছি 😊\""),
    ("নতুন কিছু শিখতে চাই, কোথা থেকে start করব?",
     "যেটা নিয়ে সত্যিই curiosity আছে সেটা বেছে নাও, রোজ মাত্র ১৫ মিনিট দাও। YouTube-এ একটা beginner playlist দিয়ে শুরু করো — consistency-ই আসল। 🚀"),
    ("office late হয়ে যাচ্ছে, boss-কে একটা quick message দাও",
     "\"Sir, রাস্তায় একটু traffic-এ আটকে গেছি, ১৫ মিনিটের মধ্যে পৌঁছে যাচ্ছি। দুঃখিত।\""),
    ("এই text-এর intent কী: ‘এই product-টা কোথায় পাব?’",
     "তথ্য-অনুসন্ধান — product কোথায় পাওয়া যাবে জানতে চাইছে।"),
]

# ---------------------------------------------------------------------------
# SLOTTED templates — thin, safe substitution (names) for volume, no quality loss.
# fmt: (instruction_template, output_template, task_type, domain)
# ---------------------------------------------------------------------------
NAMES = ["Rahul", "Priya", "Arjun", "Sneha", "Ananya", "Dev", "Riya", "Sourav",
         "Ishita", "Aritra", "Mou", "Bubai"]

SLOTTED = [
    ("{name} er birthday, ekTa short sweet Banglish wish likhe de",
     "\"Shubho jonmodin {name}! 🎂 Tor bochor ta darun kaTuk, sob swopno purno hok. "
     "Onek bhalobasa r shubhechha roilo. 🎉\"",
     "creative", "relationships"),
    ("{name} ke ekTa short congratulations message de — o notun chakri peyeche",
     "\"Congratulations {name}! 🎉 Notun chakri er jonno onek shubhechha. Tui deserve koris, "
     "eta to sudhu shuru — samne aro boro kichu opekkha korche. 🙌\"",
     "creative", "work"),
    ("{name} ke ekTa thank you message likhe de, o amay onek help korechilo",
     "\"{name}, sedin tor help ta sotti mone rakhbo. Dorkar er somoy pashe thakli — onek "
     "dhonnobad re. 🙏\"",
     "creative", "daily_life"),
    ("{name} ke dinner er jonno invite korte ekTa casual message likhe de",
     "\"Ei {name}, ei weekend e amar barite dinner e ay na! Onek din adda hoy na, eksathe "
     "kheye-daye jome jabe. Bol kobe free achis. 😄\"",
     "creative", "daily_life"),
    ("{name} ke deri howar jonno sorry bolte ekTa short message likhe de",
     "\"{name}, ajke deri kore fellam, sotti sorry. Iccha kore na, ektu atke gechilam. "
     "Ekbar janio, poshiye debo. 🙏\"",
     "creative", "daily_life"),
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _row(instruction, output, task_type, domain, script, source):
    rid = hashlib.sha1(instruction.strip().lower().encode("utf-8")).hexdigest()[:12]
    return {
        "id": rid,
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip(),
        "task_type": task_type,
        "domain": domain,
        "script": script,
        "source": source,
        "augmented": False,
    }


def build_records(names_per_template: int) -> list[dict]:
    rows: list[dict] = []

    for e in GOLD:
        rows.append(_row(e["instruction"], e["output"], e["task_type"],
                         e["domain"], e["script"], "gold"))

    for text, label, reason in BANK_SENTIMENT:
        rows.append(_row(
            f"ei text ta positive, negative na neutral bol: '{text}'",
            f"{label}. {reason.capitalize()}.",
            "classification", "daily_life", "romanized", "bank:sentiment"))

    for bn, en in BANK_TRANSLATE_EN:
        rows.append(_row(f"ei ta English e translate kore de: '{bn}'", en,
                         "translate", "daily_life", "romanized", "bank:translate_en"))

    for en, bn in BANK_TRANSLATE_BANGLISH:
        rows.append(_row(f"eita casual Banglish e bolo: '{en}'", bn,
                         "translate", "daily_life", "romanized", "bank:translate_banglish"))

    for casual, polite in BANK_REWRITE:
        rows.append(_row(f"ei line ta polite/professional kore de: '{casual}'", polite,
                         "rewrite", "work", "romanized", "bank:rewrite"))

    for messy, clean in BANK_GRAMMAR:
        rows.append(_row(f"ei banglish line er banan/gathon thik kore de: '{messy}'", clean,
                         "grammar", "daily_life", "romanized", "bank:grammar"))

    for msg, intent in BANK_INTENT:
        rows.append(_row(f"ei message er intent/uddeshyo ek katha y bol: '{msg}'", intent,
                         "intent", "daily_life", "romanized", "bank:intent"))

    for text, extracted in BANK_EXTRACT:
        rows.append(_row(f"ei text theke dorkari tottho (naam/jayga/number/somoy) ber kore de: '{text}'",
                         extracted, "extraction", "daily_life", "romanized", "bank:extract"))

    for problem, answer in BANK_MATH:
        rows.append(_row(problem, answer, "math", "education", "romanized", "bank:math"))

    for q, a in BANK_QA_FACT:
        rows.append(_row(q, a, "qa", "education", "romanized", "bank:qa_fact"))

    for req, refusal in BANK_SAFETY:
        rows.append(_row(req, refusal, "safety", "daily_life", "romanized", "bank:safety"))

    for para, summ in BANK_SUMMARIZE:
        rows.append(_row(f"ei ta ek line e summarize kor: '{para}'", summ,
                         "summarize", "daily_life", "romanized", "bank:summarize"))

    for q, a in BANK_REASONING:
        rows.append(_row(q, a, "reasoning", "education", "romanized", "bank:reasoning"))

    for ins, out in BANK_NATIVE:
        rows.append(_row(ins, out, "mixed_task", "daily_life", "native", "bank:native"))

    for ins, out in BANK_MIXED:
        rows.append(_row(ins, out, "mixed_task", "daily_life", "mixed", "bank:mixed"))

    chosen = NAMES[:max(1, names_per_template)]
    for ins_t, out_t, tt, dom in SLOTTED:
        for nm in chosen:
            rows.append(_row(ins_t.format(name=nm), out_t.format(name=nm),
                             tt, dom, "romanized", "template:slotted"))

    # de-dup on normalized instruction
    seen, deduped = set(), []
    for r in rows:
        key = re.sub(r"\s+", " ", r["instruction"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def split_records(rows: list[dict], test_frac: float) -> list[dict]:
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_test = max(1, int(len(rows) * test_frac))
    test_ids = set(idx[:n_test])
    for i, r in enumerate(rows):
        r["split"] = "heldout" if i in test_ids else "train"
    return [rows[i] for i in idx]  # return shuffled order


# ---------------------------------------------------------------------------
# Spelling-variation augmentation — the robustness multiplier.
#
# Banglish has NO standard orthography: "ache/ase/achhe", "kivabe/kemne/kmne",
# "bhalo/valo/vlo" all mean the same thing. Off-the-shelf models overfit to one
# spelling and break on the rest. We deterministically generate the real variants
# people type so the model learns the *meaning*, not the surface form.
#
# Only meaning-preserving substitutions (never person/formality changers like
# tumi<->tui). Applied to the TRAIN split ONLY — heldout stays canonical so eval
# is honest and leakage-free.
# ---------------------------------------------------------------------------
SPELLING_VARIANTS = {
    "ache": ["ase", "achhe"], "achi": ["achhi"], "achis": ["achhis"],
    "korchi": ["korchhi", "krchi"], "korche": ["korchhe"], "korbo": ["krbo"],
    "korte": ["krte"], "kore": ["kre"], "kori": ["kri"], "korbe": ["krbe"],
    "kivabe": ["kibhabe", "kemne", "kmne"], "keno": ["kno"], "kothay": ["kothae"],
    "amar": ["amr"], "amake": ["amay", "amk"], "tomake": ["tomay", "tmk"],
    "khub": ["khb", "onek"], "bhalo": ["valo", "vlo"], "valo": ["bhalo", "vlo"],
    "ekta": ["akta", "1ta"], "ektu": ["aktu", "ektuu"], "aj": ["ajke", "aaj"],
    "kal": ["kalke", "kl"], "please": ["plz", "pls"], "hobe": ["hbe"],
    "hoye": ["hoe"], "jonno": ["jonne", "jnno"], "sathe": ["shathe"],
    "dekha": ["dakha"], "hocche": ["hochche", "hocce"], "onek": ["onk"],
    "bujhte": ["bujte"], "message": ["msg"], "korle": ["krle"], "gele": ["gle"],
}


def _match_case(src: str, variant: str) -> str:
    if src.isupper():
        return variant.upper()
    if src[:1].isupper():
        return variant[:1].upper() + variant[1:]
    return variant


def augment_spelling(rows: list[dict], variants_per_row: int, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    seen = {re.sub(r"\s+", " ", r["instruction"].strip().lower()) for r in rows}
    for r in rows:
        if r["split"] != "train" or r["script"] == "native":
            continue
        tokens = re.findall(r"\w+|\W+", r["instruction"])
        elig = [i for i, t in enumerate(tokens) if t.lower() in SPELLING_VARIANTS]
        if not elig:
            continue
        made, attempts = 0, 0
        while made < variants_per_row and attempts < 8:
            attempts += 1
            new_tokens = tokens[:]
            for i in elig:
                if rng.random() < 0.6:
                    v = rng.choice(SPELLING_VARIANTS[tokens[i].lower()])
                    new_tokens[i] = _match_case(tokens[i], v)
            new_ins = "".join(new_tokens)
            key = re.sub(r"\s+", " ", new_ins.strip().lower())
            if new_ins == r["instruction"] or key in seen:
                continue
            seen.add(key)
            nr = dict(r)
            nr["instruction"] = new_ins
            nr["id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            nr["source"] = r["source"] + "+aug:spelling"
            nr["augmented"] = True
            out.append(nr)
            made += 1
    return out


# ---------------------------------------------------------------------------
# Writers + stats
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_outputs(rows: list[dict]):
    write_jsonl(DATA_DIR / "banglish_instructions.jsonl", rows)
    write_jsonl(DATA_DIR / "train.jsonl", [r for r in rows if r["split"] == "train"])
    write_jsonl(DATA_DIR / "heldout.jsonl", [r for r in rows if r["split"] == "heldout"])
    write_jsonl(DATA_DIR / "sample.jsonl", rows[:15])

    # flat CSV for the Adaption Adaptive-Data pipeline (matches adaptive_pipeline.py)
    with (DATA_DIR / "banglish_instructions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_prompt", "response"])
        for r in rows:
            w.writerow([r["instruction"], r["output"]])


def print_stats(rows: list[dict]):
    def dist(key):
        d: dict[str, int] = {}
        for r in rows:
            d[r[key]] = d.get(r[key], 0) + 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    n = len(rows)
    tr = sum(1 for r in rows if r["split"] == "train")
    he = sum(1 for r in rows if r["split"] == "heldout")
    aug = sum(1 for r in rows if r.get("augmented"))
    print(f"\nBanglaBridge dataset - {n} pairs  (train {tr} / heldout {he})")
    print(f"  authored/base: {n - aug}   spelling-augmented: {aug}")
    print("-" * 52)
    for key in ("task_type", "domain", "script"):
        print(f"\n{key}:")
        for k, v in dist(key).items():
            print(f"  {k:<22} {v:>4}  ({v / n * 100:4.1f}%)")
    print()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the BanglaBridge Banglish dataset.")
    ap.add_argument("--names-per-template", type=int, default=8,
                    help="slot-expansion breadth for templated pairs")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="held-out fraction (our internal split; real eval = Adaption's test set)")
    ap.add_argument("--aug-variants", type=int, default=1,
                    help="spelling-variant copies per eligible TRAIN row (0 = disable augmentation)")
    ap.add_argument("--stats-only", action="store_true", help="build in memory, print stats, no files")
    args = ap.parse_args()

    rows = build_records(args.names_per_template)
    rows = split_records(rows, args.test_frac)

    if args.aug_variants > 0:
        rng = random.Random(SEED + 1)
        aug = augment_spelling(rows, args.aug_variants, rng)
        rows.extend(aug)
        rng.shuffle(rows)
        print(f"Spelling augmentation: +{len(aug)} train variants (heldout left canonical)")

    if not args.stats_only:
        write_outputs(rows)
        print("Wrote: banglish_instructions.jsonl / .csv, train.jsonl, heldout.jsonl, sample.jsonl")

    print_stats(rows)


if __name__ == "__main__":
    main()
