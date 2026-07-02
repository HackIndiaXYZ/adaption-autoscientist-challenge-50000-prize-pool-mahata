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

BANK_MATH = [  # (problem, step-by-step answer)
    ("5 ta kola 30 taka hole 1 ta koto?",
     "Sohoj bhag er onko:\n- 5 ta kola = 30 taka\n- 1 ta kola = 30 ÷ 5 = **6 taka**\n\nCheck kore ne: 6 × 5 = 30 ✓"),
    ("100 takar 20% koto?",
     "Percentage er niyom: (mul × %) ÷ 100\n\n- (100 × 20) ÷ 100 = **20 taka**\n\nShortcut: 100 takar khetre % er number tai answer — 20% mane sojasuji 20 taka."),
    ("800 takar jinis e 25% chhar hole koto porbe?",
     "Step by step:\n1. Chhar = 800 er 25% = (800 × 25) ÷ 100 = 200 taka\n2. Dam porbe = 800 − 200 = **600 taka**\n\nShortcut: 25% chhar mane dam er 75% dite hobe → 800 × 0.75 = 600."),
    ("12 ar 18 er jogfol koto?",
     "12 + 18 = **30**.\n\nMone mone korar trick: 12 + 18 = 12 + 20 − 2 = 30 — round number banie jog korle onek sohoj hoy."),
    ("ekTa ghor 4 mitar * 5 mitar, khetrofol koto?",
     "Ayotokkhetrer khetrofol = dirgho × prostho\n\n- 4 × 5 = **20 borgo mitar (m²)**\n\nMone rakhis: khetrofol sob somoy 'borgo' unit e hoy — mitar × mitar = borgo mitar."),
    ("50 taka theke 35 taka kharcha hole koto baki?",
     "Biyog er onko: 50 − 35 = **15 taka** baki thakbe.\n\nCheck: 35 (kharcha) + 15 (baki) = 50 ✓"),
    ("ek dozon dim mane koto?",
     "Ek dozon = **12 ta**. Tai ek dozon dim mane 12 ta dim.\n\nBonus: 'half dozen' mane 6 ta, ar 12 dozon (144 ta) ke bole ek 'gross'."),
    ("6 jon e 300 taka soman bhag korle ke koto?",
     "Soman bhag = mot taka ÷ lok songkha\n\n- 300 ÷ 6 = **50 taka** kore protyeke pabe.\n\nCheck: 50 × 6 = 300 ✓"),
    ("ekTa kaj 3 din e hole, dwigun kaj koto din e?",
     "Ek e speed e kaj korle somoy o dwigun lagbe:\n- 1 gun kaj = 3 din\n- 2 gun kaj = 3 × 2 = **6 din**"),
    ("25 er borgo (square) koto?",
     "Borgo mane nijeke nije gun kora: 25 × 25 = **625**.\n\n5 diye sesh number er trick: ager ongko × tar porer number (2 × 3 = 6), sese 25 bosa → 625."),
    ("ekTa shirt 450 taka, duto nile koto?",
     "450 × 2 = **900 taka**.\n\nMone mone korar upay: 450 = 400 + 50 → (400 × 2) + (50 × 2) = 800 + 100 = 900."),
    ("sokal 8 ta theke bikel 2 ta koto ghonta?",
     "Dui bhag e bhenge ne:\n- Sokal 8 ta → dupur 12 ta = 4 ghonta\n- Dupur 12 ta → bikel 2 ta = 2 ghonta\n\nMot: 4 + 2 = **6 ghonta**."),
    ("200 takar bill e 15% tip koto?",
     "Tip = (200 × 15) ÷ 100 = **30 taka**\n\nMone mone shortcut: 10% = 20 taka, 5% = tar ordhek 10 taka → 15% = 20 + 10 = 30. Mot dite hobe 230 taka."),
    ("ekTa number er ordhek 40 hole number ta koto?",
     "Ulto dik theke bhab: ordhek jodi 40 hoy, pura ta tar dwigun.\n\n- Number = 40 × 2 = **80**\n\nCheck: 80 er ordhek = 40 ✓"),
    ("1 saptah = 7 din hole 3 saptah e koto din?",
     "7 din × 3 saptah = **21 din**.\n\nEi dhoroner onko sob somoy gun: mot = ek unit er maan × koyTa unit ache."),
    ("120 km 2 ghontay gele speed koto?",
     "Speed er sutra: speed = durotto ÷ somoy\n\n- 120 ÷ 2 = **60 km/ghonta**\n\nEi sutra ghurie onno duto o pabi: durotto = speed × somoy, somoy = durotto ÷ speed."),
]

BANK_QA_FACT = [  # (question, answer + ekta interesting bonus fact) — general, verifiable, safe
    ("Bharoter jatiyo pakhi ki?",
     "Bharoter jatiyo pakhi **moyur** (peacock).\n\nBonus: borshakale moyur pekhom mele nache — ei drisho Bharotiyo shilpo-sahitye bohu jayga y eseche."),
    ("Paschimbanga r rajdhani kothay?",
     "Paschimbanga r rajdhani **Kolkata**.\n\nKolkata ke 'City of Joy' o bola hoy, ar 1911 sal porjonto eta puro British India r rajdhani chilo."),
    ("Suryo purbo na paschim dike othe?",
     "Suryo **purbo** dike othe ar paschim dike ast jay.\n\nAsole suryo sthir — prithibi paschim theke purbo dike ghore bole amader chokhe suryo ke purbo dike uthte dekhi."),
    ("Jol er rasayonik sutra ki?",
     "Jol er rasayonik sutra **H₂O** — duTo hydrogen poromanu ar ekTa oxygen poromanu mile ek molecule jol hoy.\n\nEi jonnei jol ke majhe majhe 'hydrogen oxide' o bola hoy."),
    ("ek bochore koto mash?",
     "Ek bochore **12 mash**.\n\nEr modhye 7 ta mashe 31 din, 4 te te 30 din, ar February te 28 din (leap year e 29 din)."),
    ("Rongdhonu te koto rong?",
     "Rongdhonu te **7 ta rong** — lal, komla, holud, sobuj, asmani, nil, beguni.\n\nEnglish e mone rakhar trick: VIBGYOR (Violet, Indigo, Blue, Green, Yellow, Orange, Red)."),
    ("prithibir sob theke boro mohasagor kon ta?",
     "Prithibir sob theke boro mohasagor **Prosanto Mohasagor** (Pacific Ocean).\n\nEta eto boro je prithibir somosto mahadesher sthol-bhag ek sathe er modhye dhuke jete pare."),
    ("Tajmahal kothay?",
     "Tajmahal **Agra, Uttar Pradesh** e, Yamuna nodir dhare.\n\nSamrat Shah Jahan tar stri Mumtaz Mahal er smritite eta banan. Eta UNESCO World Heritage Site ar prithibir 'notun sat ascharjo'-r ekTa."),
    ("proaptobayosko manush er shorire koto ta haar thake?",
     "Proaptoboyosko manusher shorire **206 ta haar** thake.\n\nMojar byapar: sishur jonmer somoy haar thake prai 300 ta — boro howar sathe sathe onekgulo haar jure ek hoye jay."),
    ("Bharoter jatiyo phul ki?",
     "Bharoter jatiyo phul **podmo** (lotus).\n\nPodmo kada-jole fote tobu nijeke porishkar rakhe — tai eke suddhota ar gyaner protik dhora hoy."),
    ("Bangla bochorer prothom mash kon ta?",
     "Bangla bochorer prothom mash **Boishakh**.\n\nPohela Boishakh (Bangla nobo-borsho) sadharonoto 14-15 April e pore — Paschimbanga ar Bangladesh dui jaygatei boro utsob."),
    ("ek kilogram e koto gram?",
     "1 kilogram = **1000 gram**.\n\n'Kilo' upossorger mane-i hajar — tai kilometer mane 1000 mitar, kilolitre mane 1000 litre."),
    ("Everest kon deshe?",
     "Mount Everest **Nepal e**, Nepal-Tibbot (Chin) simantay.\n\nEr uchchota prai 8,849 mitar — prithibir sobcheye uchu shikhor. Nepali te er naam 'Sagarmatha'."),
    ("Ganga nodi kothay giye poreche?",
     "Ganga nodi **Bongoposagor e** (Bay of Bengal) giye poreche.\n\nPorar age Ganga onek shakhay bhag hoye jay — tar-i ekTa amader Hooghly nodi, jar dhare Kolkata."),
    ("manush ki diye shash ney?",
     "Manush **fusfus (lungs)** diye shash ney.\n\nShash e amra oxygen nei ar carbon dioxide chhari — ei binimoy ta hoy fusfuser lokkho lokkho khude thoili te, jader naam alveoli."),
    ("Bharoter swadhinota dibas kobe?",
     "Bharoter swadhinota dibas **15 August** — 1947 saler ei dine desh British shason theke swadhin hoy.\n\nProti bochor ei dine Dilli r Lal Kella y jatiyo potaka tola hoy."),
    ("suryo ki?",
     "Suryo ekTa **nokkhotro (star)** — amader sob theke kachher nokkhotro.\n\nEta eto boro je er bhitore prai 13 lokkho prithibi dhore jabe, ar er alo amader kachhe pouchate prai 8 minute 20 second lage."),
    ("ek mile e prai koto kilometer?",
     "1 mile ≈ **1.6 kilometer** (thik dhorle 1.609 km).\n\nUlto dike 1 km ≈ 0.62 mile. Bharot e amra km e mapi, kintu US/UK er rastay mile dekhbi."),
    ("chand prithibir ki?",
     "Chand prithibir **upogroho (natural satellite)** — prithibi ke ghure ghure chole.\n\nChander nijer kono alo nei; suryer alo reflect kore bole eke ujjwal dekhay. Prithibi ke ekbar ghure aste er prai 27 din lage."),
    ("jol koto degree te fote?",
     "Jol **100°C (Celsius)** e fote — tobe eta samudropristher chape.\n\nPahare uthle bataser chap kome, tai jol 100 er kom tapmatra tei fute jay — ei jonnei uchu pahare rannay beshi somoy lage."),
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

BANK_REASONING = [  # (question, answer with the reasoning chain spelled out)
    ("aj Sombar hole 3 din por ki bar?",
     "Ek din kore egiye dekh:\n- 1 din por: Mongolbar\n- 2 din por: Budhbar\n- 3 din por: **Brihospotibar** ✓"),
    ("2, 4, 6, 8 — er por ki asbe?",
     "Pattern ta dhor: 2 → 4 → 6 → 8 — protibar **+2** kore barche (jor songkhar dhara).\n\nTai porer ta: 8 + 2 = **10**."),
    ("A, C, E, G — er por kon okkhor?",
     "Pattern: A(1) → C(3) → E(5) → G(7) — majhe ekTa kore okkhor skip hocche.\n\nG er por H skip → **I** (9 number okkhor)."),
    ("Rahul er boyos Sita r dwigun. Sita 10 bochor. Rahul koto?",
     "Ja jani ta sajiye ne:\n- Sita r boyos = 10\n- Rahul = Sita r dwigun = 2 × 10 = **20 bochor**\n\nCheck: 20 ki 10 er dwigun? Hya ✓"),
    ("kolom khatar theke sosta. khata 40 taka. kolom ki 50 hote pare?",
     "Logic ta bhenge dekh:\n1. Kolom khatar theke sosta.\n2. Khata r dam 40 taka.\n3. Tahole kolomer dam 40 er **kom** hote hobe.\n\n50 > 40, tai kolom 50 taka hote **pare na**."),
    ("sob biral prani. kichu prani kalo. tahole ki sob biral kalo?",
     "Na, ei siddhanto Tana jay na. Karon:\n- 'Sob biral prani' — thik ache.\n- 'Kichu prani kalo' — kintu sei 'kichu prani'-r modhye biral nao thakte pare.\n\nTai 'sob biral kalo' bola **bhul** hobe — duTo statement theke eta proman hoy na."),
    ("ghori te ekhon 3:00. 90 minute pore koTa bajbe?",
     "90 minute = 1 ghonta 30 minute. Ebar dhape dhape:\n- 3:00 + 1 ghonta = 4:00\n- 4:00 + 30 minute = **4:30** ✓"),
    ("Ma-r ekTa meye, naam Mita. Mita r bhai er ma ke?",
     "Dhap e dhap e bhab:\n1. Bhodromohila r meye — Mita.\n2. Mita r bhai mane sei bhodromohila r-i chhele.\n3. Tahole Mita r bhai er ma **sei ek-i bhodromohila**.\n\nEi dhoroner dhadhay naam gulo confuse kore, somporko ta na — somporko dhorei egote hoy."),
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
# PREMIUM — long-form, hand-authored pairs with structured, tutorial-grade
# outputs. Weighted toward NATIVE script (Adaption's Bengali standard leans
# native). Same _g() shape as GOLD.
# ---------------------------------------------------------------------------
PREMIUM = [
    # ---- native script ----
    _g(
        "ভালো দুধ চা বানানোর নিয়মটা ধাপে ধাপে বলো",
        """পারফেক্ট দুধ চা বানানোর সহজ নিয়ম:

1. **জল ফোটাও** — ১ কাপ জল কড়া করে ফুটিয়ে নাও।
2. **চা পাতা দাও** — ফুটন্ত জলে ১.৫ চামচ চা পাতা দিয়ে ২ মিনিট ফোটাও, যাতে রংটা ভালো আসে।
3. **দুধ মেশাও** — আধ কাপ দুধ দিয়ে আরও ২–৩ মিনিট মাঝারি আঁচে ফোটাও।
4. **চিনি** — স্বাদমতো চিনি দাও; ইচ্ছে হলে এক টুকরো আদা থেঁতো করে দাও।
5. ছেঁকে গরম গরম পরিবেশন করো। ☕

টিপ: দুধ দেওয়ার পর বেশি ফোটালে চা কষা হয়ে যায় — খেয়াল রেখো।""",
        "how_to", "food", "native",
    ),
    _g(
        "আকাশ নীল দেখায় কেন? সহজ করে বুঝিয়ে দাও",
        """সহজ করে বললে:

সূর্যের সাদা আলোর মধ্যে আসলে সাত রঙের আলো মেশানো থাকে। এই আলো বাতাসের ছোট ছোট কণার গায়ে ধাক্কা খেলে সব রং সমানভাবে ছড়ায় না — **নীল আলো সবচেয়ে বেশি ছড়িয়ে পড়ে** (এর তরঙ্গদৈর্ঘ্য ছোট বলে)।

চারদিকে ছড়িয়ে পড়া সেই নীল আলোই আমাদের চোখে আসে, তাই আকাশ নীল দেখায়।

মজার ব্যাপার: সূর্যাস্তের সময় আলোকে অনেকটা বেশি বাতাস পেরোতে হয় — নীল আগেই ছড়িয়ে হারিয়ে যায়, পড়ে থাকে লাল-কমলা। তাই সূর্যাস্ত লাল। 🌅""",
        "explain", "education", "native",
    ),
    _g(
        "অফিসে তিন দিনের ছুটির জন্য একটা দরখাস্ত লিখে দাও",
        """বিষয়: তিন দিনের ছুটির আবেদন

মহাশয়,

সবিনয়ে জানাই, পারিবারিক প্রয়োজনে আগামী [তারিখ] থেকে [তারিখ] পর্যন্ত মোট তিন দিন আমি অফিসে উপস্থিত থাকতে পারব না। এই সময়ের জরুরি কাজগুলো আমি আগেই গুছিয়ে দিয়ে যাব, এবং প্রয়োজনে ফোন বা ই-মেলে আমাকে পাওয়া যাবে।

অনুগ্রহ করে ছুটি মঞ্জুর করলে বাধিত থাকব।

ধন্যবাদান্তে,
[আপনার নাম]
[পদ, বিভাগ]
[তারিখ]""",
        "writing", "work", "native",
    ),
    _g(
        "পরীক্ষার আগে এক মাসের পড়ার রুটিন কীভাবে বানাব?",
        """এক মাসের পড়ার রুটিন বানানোর সহজ পদ্ধতি:

**সপ্তাহ ১–২: বুঝে পড়া**
- সিলেবাস ভাগ করো — কোন বিষয়ে কত অধ্যায়, লিখে ফেলো।
- কঠিন বিষয় দিনের শুরুতে রাখো, যখন মাথা সবচেয়ে ফ্রেশ।

**সপ্তাহ ৩: লেখা প্র্যাকটিস**
- পুরনো প্রশ্নপত্র ধরে সময় মেপে লেখো।
- ভুলগুলো আলাদা খাতায় টুকে রাখো — এটাই তোমার আসল নোট।

**সপ্তাহ ৪: রিভিশন**
- নতুন কিছু নয়, শুধু ঝালিয়ে নেওয়া।
- রোজ ৭–৮ ঘণ্টা ঘুম — মুখস্থ টেকানোর জন্য ঘুম জরুরি।

টিপ: টানা ২–৩ ঘণ্টা নয় — ৪৫ মিনিট পড়া + ১০ মিনিট বিরতি, এতে মনোযোগ অনেক বেশি থাকে। 📚""",
        "how_to", "education", "native",
    ),
    _g(
        "AI মানে কী? একদম সহজ ভাষায় বোঝাও",
        """AI (Artificial Intelligence) বা কৃত্রিম বুদ্ধিমত্তা মানে — **কম্পিউটারকে এমনভাবে শেখানো, যাতে সে মানুষের মতো বুদ্ধির কাজ করতে পারে**।

যেমন:
- ছবি দেখে বলা এটা বেড়াল না কুকুর
- তোমার কথা বুঝে উত্তর দেওয়া (যেমন এই চ্যাট!)
- এক ভাষা থেকে আরেক ভাষায় অনুবাদ

কীভাবে শেখে? হাজার হাজার উদাহরণ দেখে। বাচ্চাকে অনেক বেড়ালের ছবি দেখালে সে যেমন বেড়াল চিনতে শেখে, AI-ও লক্ষ লক্ষ উদাহরণ থেকে প্যাটার্ন শিখে নেয়।

মনে রেখো: AI জাদু নয় — অঙ্ক আর ডেটার খেলা।""",
        "explain", "tech", "native",
    ),
    _g(
        "ফোনের স্টোরেজ বারবার ভরে যাচ্ছে, কী করব?",
        """ফোনের স্টোরেজ খালি করার কাজের টিপস:

1. **WhatsApp মিডিয়া** — সবচেয়ে বড় অপরাধী! Settings → Storage থেকে বড় ফাইল আর ফালতু ভিডিও ডিলিট করো।
2. **ছবি ব্যাকআপ** — Google Photos-এ ব্যাকআপ নিয়ে 'Free up space' চাপো — ফোন থেকে কপি মুছবে, ক্লাউডে থেকে যাবে।
3. **অব্যবহৃত অ্যাপ** — ৩ মাস খোলোনি এমন অ্যাপ আনইনস্টল করো।
4. **Cache পরিষ্কার** — Settings → Apps → বড় অ্যাপগুলোর cache clear করো।

এই চারটে করলেই সাধারণত ৫–১০ GB খালি হয়ে যায়। 📱""",
        "how_to", "tech", "native",
    ),
    _g(
        "মাসের শেষে টাকা থাকে না, কীভাবে সেভ করা শুরু করব?",
        """টাকা জমানো শুরু করার বাস্তব উপায়:

1. **আগে জমাও, পরে খরচ** — মাইনে পেয়েই ১০–২০% আলাদা অ্যাকাউন্টে সরিয়ে ফেলো। যা থাকে তা দিয়ে মাস চালাও — উল্টোটা নয়।
2. **খরচ লেখো** — এক মাস প্রতিটা খরচ টুকে দেখো। টাকা কোথায় 'হারিয়ে যাচ্ছে' দেখে নিজেই চমকে যাবে (বেশিরভাগ সময়: বাইরের খাবার আর সাবস্ক্রিপশন)।
3. **৫০-৩০-২০ নিয়ম** — আয়ের ৫০% প্রয়োজনে, ৩০% শখে, ২০% সঞ্চয়ে।
4. ছোট শুরুই যথেষ্ট — দিনে ৫০ টাকা মানেও বছরে ১৮,০০০+।

মনে রেখো: জমানো অভ্যাসের ব্যাপার, অঙ্কের নয়। 💰""",
        "advice", "daily_life", "native",
    ),
    _g(
        "রাতে ঘুম আসে না, কী করতে পারি?",
        """ভালো ঘুমের জন্য পরীক্ষিত কিছু অভ্যাস:

1. **স্ক্রিন বন্ধ** — শোয়ার অন্তত ৩০–৬০ মিনিট আগে ফোন রেখে দাও; স্ক্রিনের আলো ঘুমের হরমোন (মেলাটোনিন) কমিয়ে দেয়।
2. **নির্দিষ্ট সময়** — রোজ একই সময়ে শোয়া-ওঠা, ছুটির দিনেও। শরীরের ঘড়ি এতে সেট হয়ে যায়।
3. **দুপুরের পর চা-কফি নয়** — ক্যাফেইন ৬+ ঘণ্টা শরীরে থেকে যায়।
4. **বিছানা শুধু ঘুমের জন্য** — বিছানায় শুয়ে কাজ বা স্ক্রল নয়।
5. ঘুম না এলে ছটফট না করে উঠে অল্প আলোয় একটু বই পড়ো, ঘুম পেলে ফেরো।

কয়েক সপ্তাহ নিয়ম মেনেও না কমলে একবার ডাক্তার দেখিয়ে নেওয়া ভালো। 😴""",
        "advice", "health", "native",
    ),
    _g(
        "জন্মদিনের নিমন্ত্রণের একটা সুন্দর মেসেজ লিখে দাও",
        """একটা আন্তরিক নিমন্ত্রণ মেসেজ:

"প্রিয় [নাম],

আগামী [তারিখ], [বার] আমার জন্মদিন উপলক্ষে বাড়িতে ছোট্ট একটা আয়োজন করছি। তুমি এলে দিনটা আরও আনন্দের হয়ে উঠবে।

সময়: সন্ধ্যা ৭টা
ঠিকানা: [ঠিকানা]

অবশ্যই এসো কিন্তু — তোমার অপেক্ষায় থাকব! 🎂"

চাইলে শেষে 'পরিবারের সবাইকে নিয়ে এসো' জুড়ে দিতে পারো — আরও আন্তরিক শোনায়।""",
        "writing", "relationships", "native",
    ),
    _g(
        "ইন্টারনেট আসলে কীভাবে কাজ করে? সহজ ভাষায়",
        """সহজ ভাষায় ইন্টারনেট:

ইন্টারনেট হলো **পৃথিবীজুড়ে কোটি কোটি কম্পিউটারের জাল**, যারা তার (এমনকি সমুদ্রের নিচের কেবল!) দিয়ে জোড়া।

তুমি যখন কোনো ওয়েবসাইট খোলো:
1. তোমার ফোন একটা 'চিঠি' (request) পাঠায় — "আমাকে এই পেজটা দাও।"
2. চিঠিটা ছোট ছোট টুকরোয় (packet) ভেঙে বহু রাউটার ঘুরে পৌঁছয় সেই সাইটের সার্ভারে — যা হয়তো অন্য দেশে।
3. সার্ভার উত্তরে পেজের ডেটা পাঠায়; টুকরোগুলো তোমার ফোনে এসে আবার জোড়া লাগে।

আর এই পুরো যাতায়াতটা হয় চোখের পলকের চেয়েও কম সময়ে। 🌐""",
        "explain", "tech", "native",
    ),
    _g(
        "চাকরির ইন্টারভিউতে খুব নার্ভাস লাগে, কী করব?",
        """ইন্টারভিউর ভয় কমানোর কার্যকর উপায়:

**আগের দিন:**
- কোম্পানি সম্পর্কে ১৫ মিনিট পড়ে যাও — 'আমাদের সম্পর্কে কী জানো?' প্রশ্নটা প্রায় সবাই করে।
- নিজের পরিচয় ২ মিনিটে বলার প্র্যাকটিস করো — আয়নার সামনে, জোরে জোরে।

**সেই দিন:**
- ১০ মিনিট আগে পৌঁছাও — তাড়াহুড়ো নার্ভাসনেস দ্বিগুণ করে।
- ঢোকার আগে ৪ সেকেন্ড শ্বাস নাও, ৪ সেকেন্ড ছাড়ো — এইভাবে ৩ বার।

**ভেতরে:**
- উত্তর না জানলে সোজা বলো: "এটা এখনই জানি না, তবে শিখে নেব।" বানানো গল্পের চেয়ে এটা অনেক ভালো দেখায়।

মনে রেখো: ওরা তোমার শত্রু নয় — ওরাও চায় প্রার্থী ভালো হোক, যাতে ওদের খোঁজা শেষ হয়। 💼""",
        "advice", "work", "native",
    ),
    _g(
        "ফ্রেশারদের CV-তে কী কী রাখা উচিত?",
        """ফ্রেশার CV-র জরুরি অংশগুলো (সব মিলিয়ে এক পাতায়!):

1. **যোগাযোগ** — নাম, ফোন, ই-মেল, LinkedIn। ই-মেলটা প্রফেশনাল রাখো (cooldude123 নয় 😅)।
2. **সামারি (২–৩ লাইন)** — কে তুমি, কী পারো, কী খুঁজছ।
3. **স্কিল** — যেগুলো সত্যিই পারো শুধু সেগুলো; ইন্টারভিউতে ধরা পড়ে যায়।
4. **প্রজেক্ট** — ফ্রেশারের আসল সম্পদ: কী বানিয়েছ + কী দিয়ে + লিংক।
5. **শিক্ষা** — ডিগ্রি, প্রতিষ্ঠান, সাল, নম্বর।
6. ইন্টার্নশিপ বা সার্টিফিকেট থাকলে অবশ্যই দাও।

যে ভুলগুলো এড়াবে: বানান ভুল, দু'পাতার বেশি লম্বা CV, আর সব জায়গায় একই CV পাঠানো — চাকরি বুঝে একটু বদলে নাও।""",
        "advice", "work", "native",
    ),
    _g(
        "বৃষ্টি কীভাবে হয়? ধাপে ধাপে বোঝাও",
        """বৃষ্টির পুরো চক্রটা চারটে ধাপে:

1. **বাষ্পীভবন** — সূর্যের তাপে নদী-সমুদ্রের জল বাষ্প হয়ে উপরে ওঠে।
2. **ঘনীভবন** — উপরের ঠান্ডা বাতাসে বাষ্প জমে ছোট ছোট জলকণা হয় — এই কোটি কোটি কণা মিলেই মেঘ।
3. **মেঘ ভারী হওয়া** — কণাগুলো জুড়ে জুড়ে বড় ফোঁটা হয়; বাতাস আর ধরে রাখতে পারে না।
4. **বৃষ্টি** — ফোঁটাগুলো নেমে আসে। ⛈️

তারপর সেই জল আবার নদী-সমুদ্রে ফিরে যায়, আবার বাষ্প হয় — এই ঘূর্ণিটার নামই **জলচক্র (water cycle)**।""",
        "explain", "education", "native",
    ),
    _g(
        "পুরসভায় রাস্তার আলো খারাপ নিয়ে একটা অভিযোগপত্র লিখে দাও",
        """বিষয়: রাস্তার আলো মেরামতের আবেদন

মাননীয় মহাশয়/মহাশয়া,

সবিনয়ে জানাই, [এলাকার নাম]-এর [রাস্তার নাম] সংলগ্ন রাস্তার আলোগুলি গত [সময়কাল] ধরে অকেজো হয়ে আছে। সন্ধ্যার পর পুরো রাস্তা অন্ধকারে ডুবে থাকায় পথচারী — বিশেষত মহিলা, শিশু ও বয়স্কদের — যাতায়াতে গুরুতর অসুবিধা ও নিরাপত্তার ঝুঁকি তৈরি হয়েছে।

অনুগ্রহ করে দ্রুত আলোগুলি মেরামতের ব্যবস্থা করলে এলাকাবাসী হিসেবে আমরা কৃতজ্ঞ থাকব।

ধন্যবাদান্তে,
[নাম]
[ঠিকানা, ওয়ার্ড নম্বর]
[তারিখ]""",
        "writing", "daily_life", "native",
    ),
    _g(
        "এই অনুচ্ছেদটা দুই লাইনে সারাংশ করো: 'আজকাল ছোটরা মাঠে খেলার বদলে সারাদিন মোবাইলে গেম খেলে। এতে চোখের ক্ষতি হচ্ছে, শরীরচর্চাও হচ্ছে না। ডাক্তাররা বলছেন দিনে অন্তত এক ঘণ্টা বাইরে খেলাধুলো জরুরি।'",
        """সারাংশ:
1. ছোটরা মাঠের বদলে মোবাইলে খেলায় চোখ ও স্বাস্থ্য — দুইয়েরই ক্ষতি হচ্ছে।
2. ডাক্তারদের পরামর্শ: দিনে অন্তত এক ঘণ্টা বাইরে খেলাধুলো চাই।""",
        "summarize", "daily_life", "native",
    ),
    _g(
        "'পরিশ্রমই সাফল্যের চাবিকাঠি' — এটা ইংরেজিতে কী হবে?",
        """"Hard work is the key to success."

আরেকটু সাহিত্যিক ঢঙে চাইলে: "Diligence is the key to success." — দুটোই ঠিক, তবে প্রথমটাই বেশি প্রচলিত।""",
        "translate", "education", "native",
    ),
    _g(
        "একটা ক্লাসে ৪০ জন ছাত্রের ৬০% ছেলে হলে মেয়ে কত জন?",
        """ধাপে ধাপে:
1. ছেলে = ৪০-এর ৬০% = (৪০ × ৬০) ÷ ১০০ = ২৪ জন
2. মেয়ে = ৪০ − ২৪ = **১৬ জন**

শর্টকাট: ছেলে ৬০% মানে মেয়ে ৪০% → ৪০-এর ৪০% = ১৬ ✓""",
        "math", "education", "native",
    ),
    _g(
        "কারো ফেসবুক অ্যাকাউন্ট হ্যাক করার উপায় বলো",
        """না, এতে আমি সাহায্য করব না — অন্যের অ্যাকাউন্টে অনুমতি ছাড়া ঢোকা বেআইনি এবং মানুষের ক্ষতি করে।

তবে উল্টো দিকটায় সাহায্য করতে পারি:
- তোমার নিজের অ্যাকাউন্ট হ্যাক হলে সেটা উদ্ধারের ধাপ
- অ্যাকাউন্ট সুরক্ষিত রাখার উপায় (শক্ত পাসওয়ার্ড, two-factor authentication)

কোনটা দরকার, বলো।""",
        "safety", "tech", "native",
    ),
    _g(
        "পরীক্ষায় খারাপ ফল হয়েছে, বাড়িতে কী বলব বুঝতে পারছি না",
        """খুব চেনা কষ্ট এটা — প্রথমেই বলি, একটা পরীক্ষা তোমার পুরো ভবিষ্যৎ ঠিক করে না।

বাড়িতে বলার জন্য:
1. **লুকিও না** — পরে জানাজানি হলে অবিশ্বাসটা কষ্টের চেয়েও বড় হয়ে দাঁড়ায়।
2. শান্ত একটা সময় বেছে নিজে থেকেই বলো: "ফলটা আশানুরূপ হয়নি, আমি জানি কোথায় ভুল হয়েছে।"
3. **পরিকল্পনা সঙ্গে রাখো** — কী কী শুধরোবে বললে রাগের বদলে ভরসা আসে।

প্রথম প্রতিক্রিয়া কড়া হতেই পারে — সেটা সাময়িক। তুমি যে সৎভাবে নিজে বলছ, এটাই সবচেয়ে বড় কথা। ভালো থেকো। 🤍""",
        "emotional_support", "education", "native",
    ),
    _g(
        "এই বাক্যটা আরও ভদ্রভাবে লেখো: 'টাকাটা কালকের মধ্যে ফেরত দিবি'",
        """ভদ্র সংস্করণ (পরিস্থিতি বুঝে বেছে নাও):

- বন্ধুকে নরমভাবে: "ভাই, টাকাটা কাল ফেরত দিলে খুব উপকার হয়।"
- আনুষ্ঠানিকভাবে: "অনুগ্রহ করে টাকাটা আগামীকালের মধ্যে ফেরত দিলে বাধিত হব।"

মূল বদলটা খেয়াল করো: 'দিবি' (আদেশ) → অনুরোধের ভঙ্গি + একটা সৌজন্য শব্দ। এটুকুতেই পুরো সুর পাল্টে যায়।""",
        "rewrite", "daily_life", "native",
    ),
    # ---- mixed code-switch ----
    _g(
        "কাল interview আছে, last minute-এ কী কী prepare করব?",
        """শেষ মুহূর্তের preparation checklist:

1. **Self-introduction** — ২ মিনিটের intro একবার জোরে বলে প্র্যাকটিস করো।
2. **Company research** — ওদের website আর LinkedIn ১৫ মিনিট দেখে নাও।
3. **নিজের CV** — যা লিখেছ সব নিয়ে প্রশ্ন হতে পারে, একবার পড়ে যাও।
4. **Documents** — CV-র কপি, ID, certificate — আজ রাতেই ব্যাগে গুছিয়ে রাখো।
5. রাত জেগো না — ঘুম ভালো হলে মাথা sharp থাকবে।

All the best! 💪""",
        "advice", "work", "mixed",
    ),
    _g(
        "phone e khub distraction hoy, পড়ায় focus করার উপায় বলো",
        """Focus ফেরানোর কাজের উপায়:

1. **Phone অন্য ঘরে** — silent নয়, একেবারে অন্য ঘরে। ফোন চোখের সামনে থাকলেই মনোযোগ কমে।
2. **Pomodoro technique** — ২৫ মিনিট পড়া + ৫ মিনিট বিরতি; ৪ রাউন্ড হলে একটা বড় বিরতি।
3. **Notification বন্ধ** — পড়ার সময়টুকু DND চালিয়ে দাও।
4. বসার আগে টেবিলে জল আর দরকারি জিনিস রেখে দাও — ওঠার অজুহাত কমবে।

প্রথম ২ দিন কষ্ট হবে, তারপর অভ্যাসে দাঁড়িয়ে যাবে। 📵""",
        "advice", "education", "mixed",
    ),
    _g(
        "amar laptop khub slow hoye geche, কী করলে fast হবে?",
        """Laptop fast করার ধাপগুলো (টাকা খরচ ছাড়াই):

1. **Startup apps বন্ধ** — Task Manager → Startup tab → অদরকারিগুলো disable করো। সবচেয়ে বেশি কাজ দেয় এটাই।
2. **Storage খালি** — C drive-এ অন্তত ১৫–২০% জায়গা ফাঁকা রাখো; Downloads folder পরিষ্কার করো।
3. **Browser tabs** — ২০টা tab মানেই RAM শেষ। দরকারিগুলো bookmark করে বাকি বন্ধ।
4. **Restart** — সপ্তাহে অন্তত একবার পুরো shutdown/restart।

এর পরেও slow লাগলে RAM বাড়ানো বা SSD লাগানো — এই দুটো upgrade-এ পুরনো laptop-ও প্রায় নতুনের মতো চলে। 💻""",
        "how_to", "tech", "mixed",
    ),
    _g(
        "ei weekend e Darjeeling jachhi, ২ দিনের plan suggest koro",
        """দার্জিলিং ২ দিনের plan:

**দিন ১:**
- সকালে পৌঁছে hotel check-in, তারপর **Mall Road** আর আশপাশ হেঁটে ঘোরো।
- দুপুরে local খাবার — মোমো আর থুকপা must! 🥟
- বিকেলে **Batasia Loop** আর **Peace Pagoda**।
- সন্ধ্যায় Mall-এ বসে গরম কফি — পাহাড়ের সন্ধ্যা এমনিই সুন্দর।

**দিন ২:**
- ভোর ৩:৩০-এ উঠে **Tiger Hill** — কাঞ্চনজঙ্ঘার সূর্যোদয়, পুরো trip-এর সেরা মুহূর্ত।
- ফেরার পথে **Ghoom Monastery**।
- সময় থাকলে **Toy Train** joyride (আগে থেকে book করা ভালো)।

টিপ: গরম জামা নিও, আর weekend-এ হোটেল আগেভাগে book করে যেও।""",
        "planning", "travel", "mixed",
    ),
    _g(
        "online shopping e fraud theke bachar upay ki? কয়েকটা টিপস দাও",
        """Online fraud থেকে বাঁচার জরুরি নিয়ম:

1. **অচেনা link-এ ক্লিক নয়** — SMS/WhatsApp-এ আসা 'অবিশ্বাস্য offer'-এর লিংক এড়িয়ে যাও; আসল সাইটে নিজে টাইপ করে ঢোকো।
2. **OTP কাউকে নয়** — ব্যাংক, delivery boy, 'customer care' — ফোনে কেউ OTP চাইলে সেটা ১০০% fraud।
3. **নতুন সাইটে Cash on Delivery** — সন্দেহ হলে আগে টাকা দিও না।
4. **URL দেখো** — https আছে কি না, বানান ঠিক কি না (flipkart vs fllpkart)।
5. ঠকে গেলে সঙ্গে সঙ্গে **1930**-এ ফোন করো (জাতীয় সাইবার ক্রাইম হেল্পলাইন) — যত দ্রুত জানাবে, টাকা ফেরার সম্ভাবনা তত বেশি। 🛡️""",
        "advice", "tech", "mixed",
    ),
    _g(
        "breakfast e healthy কিন্তু tasty কী বানানো যায়? ২-৩টে idea দাও",
        """সহজ, healthy আর সত্যিই খেতে ভালো — এই তিনটে try করো:

1. **চিঁড়ের পোলাও / পোহা** — চিঁড়ে + পেঁয়াজ + বাদাম + কারিপাতা। ১০ মিনিটে হয়, হালকা অথচ পেট ভরে।
2. **ডিম-সবজি টোস্ট** — brown bread-এ ডিম ভাজা + টমেটো-শসা। প্রোটিন আর ফাইবার একসাথে।
3. **ঝাল ওটস উপমা** — মিষ্টি ওটস ভালো না লাগলে এটা game-changer: ওটস + সবজি + সর্ষে ফোড়ন।

সঙ্গে একটা ফল (কলা/আপেল) রাখলে ষোলো আনা। সকালে প্রোটিন থাকলে সারাদিন খিদেও কম পায়, energy-ও থাকে। 🍳""",
        "how_to", "food", "mixed",
    ),
    # ---- romanized long-form ----
    _g(
        "UPI diye bhul number e taka pathiye felechi, ki korbo ekhon?",
        """Tension nis na — step by step ja korbi:

1. **Sathe sathe proman gochha** — transaction ID, somoy, amount er screenshot ne.
2. **UPI app er Help e complaint kor** — GPay/PhonePe/Paytm sob app ei 'payment sent to wrong account' option ache.
3. **Nijer bank er customer care e phone kor** — complaint number ta likhe rakh, follow-up e lagbe.
4. Receiver ke chinle bhodro bhabe request kor — onek manush emni tei ferot dey.
5. Boro amount hole bank e giye written complaint kor.

Sob theke important: **joto taratari report korbi, taka ferot pawar chance toto beshi.** 💸""",
        "how_to", "tech",
    ),
    _g(
        "roj 30 minute somoy pai, English bolar practice kivabe korbo?",
        """Roj 30 minute er ekTa kaje-lagar moto plan:

**10 min — Shona:** English podcast ba YouTube (subtitle soho). Prothome slow speed e shon.
**10 min — Bola:** Aynar samne aj sara din ki korli, seTa English e bol. Bhul hobe — thamis na. Fluency age ase, grammar pore.
**10 min — Notun shobdo:** Roj 5 ta notun word, protiTar sathe nijer jiboner ekTa sentence banie likh.

Extra boost:
- Phone er language English kore de — sara din free practice.
- Bondhur sathe din e 15 minute 'English only' rule kor — dujonei shikhbi.

3 mash regular korle nijei difference Ter pabi. Talent na, **consistency** i asol. 🗣️""",
        "how_to", "education",
    ),
    _g(
        "monthly budget kivabe banabo? amar salary 25000",
        """25,000 takar ekTa bastob budget (50-30-20 rule):

**Proyojon — 12,500 (50%)**
- Bari bhara / mess: ~7,000
- Khawa-dawa: ~4,000
- Jatayat + recharge: ~1,500

**Nijer shokh — 7,500 (30%)**
- Bondhu-bandhob, cinema, shopping, bairer khabar

**Sanchoy — 5,000 (20%)**
- Salary dhukei prothome ei 5,000 alada account/RD te sorie fel
- 3-4 mash e ekTa emergency fund dnariye gele dekhbi tension koto kome

Nijer moto adjust kor — bhara beshi hole shokher bhag theke kaT, sanchoy theke na. **Sanchoy holo nijeke deya salary.** 💰""",
        "planning", "daily_life",
    ),
    _g(
        "second hand bike kenar age ki ki dekhe nebo?",
        """Second-hand bike kenar age must-check list:

**Kagoj-potro:**
1. RC book — malik er naam ar engine/chassis number bike er sathe milie dekh.
2. Insurance valid ache ki na.
3. Onno rajyer bike hole NOC lagbe.

**Bike er obostha:**
4. **Thanda engine e** start kor — ek chape start hoy ki na. Kalo/sada dhnoya berole somossa ache.
5. Odometer er sathe bike er chehara milche ki na — 20,000 km lekha othocho seat-handle khoye gechhe mane meter ghorano hote pare.
6. Chain, brake, tyre er khoy dekh.
7. Test ride ne — gear smooth porche ki na, ajob kono awaj hocche ki na.

Sesh advice: mechanic bondhu thakle sathe nie ja — or 10 minute er chokh tor 10-15 hajar taka bnachate pare. 🏍️""",
        "advice", "daily_life",
    ),
    _g(
        "ma-baba ke kivabe bojhabo je ami engineering na, design porte chai?",
        """Ei conversation Ta kothin, kintu plan kore korle onek sohoj hoye jay:

1. **Homework kore ja** — design e ki ki career (UI/UX, product, graphic), starting salary koto, kon college/course bhalo — concrete tothyo nie bosh. 'Bhalo lage' er cheye '**ei field e ei ei sujog ache**' onek beshi kaj kore.
2. **Thik somoy bechhe ne** — sobai shanto emon somoy; kono jhogra r por na.
3. **Tader bhoy Ta age shikar kor** — bol: 'Ami jani tomra amar bhobishyot nie chinta koro.' Ei ek line ei orddhek deyal pore jay.
4. **Kaj dekha** — nijer design er portfolio, kono competition ba certificate thakle samne rakh. Kotha r cheye kaj beshi bojhay.
5. Ek din e 'hya' asbe na — bij pnute de, somoy de.

Mone rakhis: tader 'na' Ta obishwas theke na, **bhoy theke ase**. Dhoirjo rakh. 🤍""",
        "advice", "relationships",
    ),
    _g(
        "ei email ta ektu polish kore de: 'sir ami interview er jonno apply korte chai. amar resume attach korlam. reply diben.'",
        """Polished version:

---
Subject: Application for [Position Name] — [Tomar Naam]

Dear Sir/Madam,

I am writing to express my interest in the [Position Name] role at [Company Name]. I have attached my resume for your kind consideration.

I would be grateful for an opportunity to discuss how my skills align with this position. Please let me know a convenient time.

Thank you for your time.

Sincerely,
[Naam]
[Phone] | [Email]
---

Ki ki bodlalo kheyal kor:
- 'reply diben' (order er moto shonay) → 'I would be grateful…' (bhodro request)
- Subject line add holo — HR er inbox e eTa-i prothome chokhe pore
- Position ar company-r naam bosano — jate copy-paste feel na ase.""",
        "rewrite", "work",
    ),
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

    for e in PREMIUM:
        rows.append(_row(e["instruction"], e["output"], e["task_type"],
                         e["domain"], e["script"], "gold:premium"))

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
