
*** 1. INTRODUCTION ***
/*
Description: 	Creates panel dataset of married women aged 16-49
Uses: 			Raw survey data,  household_allwaves_master.dta
Creates: 		marriedwoman16to49_allwaves_master.dta
*/

*** 2. SET ENVIRONMENT ***

version 14.1 		
set more off		
clear all			
pause on 			

//Prompt for directory if not filled in above
if "$PKH" == "" {
	display "Enter the directory where the PKH files can be located: " _request(PKH)
	* add a \ if they forgot one
	quietly if substr("$PKH",length("PKH"),1) != "/" & substr("$PKH",length("$PKH"),1) != "\" {
		global PKH = "$PKH/"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local TARGET2 = "$TARGET2"

cd "`PKH'"

//Set working directory:
cd "`pkhdata'"

*** 3. CODING HEALTH OUTCOMES BY WAVE ***
*Wave I
//code outcomes from pregnancy record file 
use "Wave I/Data/I_CH_fin", clear
duplicates report // no duplicates

//coding pre-natal visits 
//do not include traditional birth visits (ich08g/ich10g/ich12g) or "other" (ich08v/ich10v/ich12v)
local prenatal ich08a ich08b ich08c ich08d ich08e ich08f ich08h ///
 			   ich10a ich10b ich10c ich10d ich10e ich10f ich10h ///
 			   ich12a ich12b ich12c ich12d ich12e ich12f ich12h 

//Recode missings as 0s
foreach x of var `prenatal' {
	replace `x' = 0 if `x' == . & ich07 != .
}

//calculate total number of pre-natal visits
egen pre_natal_visits = rowtotal(`prenatal'), missing 
replace pre_natal_visits = 20 if pre_natal_visits > 20 & pre_natal_visits < . //top-code unreasonable counts 

//coding dummy for >90 iron pills
gen iron_pills_dummy = 1 if ich21 == 4
replace iron_pills_dummy = 0 if ich21 < 4

gen iron_pills_categ = ich21
replace iron_pills_categ = 0 if ich20 == 3
label define ironcat 0 "Did not receive any" 1 "1-30 pills" 2 "31-60 pills" 3 "61-90 pills" 4 ">90 pills" 8 "Do Not Know, but did receive"
label values iron_pills_categ ironcat
replace iron_pills_categ = . if ich21 == 8


//coding assisted delivery
//generate dummy if doctor assisted in delivery 
gen dummy_A = strmatch(ich25, "*A*")
replace dummy_A = . if missing(ich25)
//dummy for midwife assisted delivery 
gen dummy_B = strmatch(ich25, "*B*")
replace dummy_B = . if missing(ich25)
//generate indicator for good assisted delivery 
gen good_assisted_delivery = (dummy_A == 1 | dummy_B == 1) if !missing(ich25)


//coding delivery at facility
gen delivery_facility = (ich24 < 7)
replace delivery_facility = . if missing(ich24)


//coding post-natal visits 
//do not include traditional birth visits (ich32g/ich33g) or "other" (ich32v/ich33v)
local postnatal ich32a ich32b ich32c ich32d ich32e ich32f ich32h ///
 				ich33a ich33b ich33c ich33d ich33e ich33f ich33h 
//Recode missings as 0s
foreach x of var `postnatal' {
	replace `x' = 0 if `x'== . & ich31 != .
}

//calculate total number of post-natal visits 
egen post_natal_visits = rowtotal(`postnatal'), missing 
replace post_natal_visits = 10 if post_natal_visits > 10 & post_natal_visits < . //top-code unreasonable counts



//birthweight missing indicator as well as actual weight
//missing indicator only for live births 
gen missing_birthweight = (missing(ich30)) if ich02 == 4
gen knows_birthweight = 1 - missing_birthweight if ich02 == 4
gen birthweight = ich30 if ich02 == 4

//coding miscarriage and miscarriage/stillbirth 
gen miscarriage = (ich02 == 2) if !missing(ich02) & ich02 != 1
gen miscarriage_or_stillborn = (ich02 == 2 | ich02 == 3) if !missing(ich02) & ich02 != 1

tempfile outcomes_w1
save `outcomes_w1', replace 

use "Wave I/Data/I_fin", clear
duplicates report iid //no duplicates

//Code knowledge on breastfeeding and diarrhea (for women with kids under age 3 years)
gen q1 = 1 if ipk01_n >= 4 & ipk01_n < .
replace q1 = 0 if ipk01 == 8 | ipk01_n < 4

gen q2 = (ipk02 == 1)
gen q3 = (ipk03_n == 180)
gen q4 = (ipk04 == 1)
gen q5 = (ipk05 == 2)

egen q_sum = rowtotal(q1 q2 q3 q4 q5)
gen mother_knowledge = q_sum / 5

//indicator for pre-natal visits question correct 
rename q1 pre_natal_question_correct  

//composite breastfeeding knowledge indicator 
egen breastfeed_correct_num = rowtotal(q2 q3)
gen breastfeed_correct_pct = breastfeed_correct_num / 2
gen breastfeed_correct_all = (breastfeed_correct_pct == 1)

//composite diarrhea knowledge indicator
egen diarrhea_correct_num = rowtotal(q4 q5)
gen diarrhea_correct_pct = diarrhea_correct_num / 2
gen diarrhea_correct_all = (diarrhea_correct_pct == 1)


//age-restricted knowledge outcomes 
gen mother_knowledge_35 = mother_knowledge if iir01 < 35
gen breastfeed_correct_pct_35 = breastfeed_correct_pct if iir01 < 35
gen diarrhea_correct_pct_35 = diarrhea_correct_pct if iir01 < 35
gen pre_natal_question_correct_35 = pre_natal_question_correct if iir01 < 35


//empowerment variables 
gen mother_inv_education = regexm(isp01a, "A") if !missing(isp01a) & isp01a != "W" // do not consider "N/A"
gen mother_excl_education = (isp01a == "A") if !missing(isp01a) & isp01a != "W"

gen mother_inv_health = regexm(isp01b, "A") if !missing(isp01b) & isp01b != "W" // do not consider "N/A"
gen mother_excl_health = (isp01b == "A") if !missing(isp01b) & isp01b != "W"

gen mother_inv_discipline = regexm(isp01c, "A") if !missing(isp01c) & isp01c != "W" // do not consider "N/A"
gen mother_excl_discipline = (isp01c == "A") if !missing(isp01c) & isp01c != "W"

gen mother_inv_birth = regexm(isp01d, "A") if !missing(isp01d) & isp01d != "W" // do not consider "N/A"
gen mother_excl_birth = (isp01d == "A") if !missing(isp01d) & isp01d != "W"

//mother has to ask permission to buy items 
gen mother_perm_buy_veg = (isp02a == 1) if !missing(isp02a) & isp02a != 6
gen mother_perm_buy_clothing = (isp02b == 1) if !missing(isp02b) & isp02b != 6
gen mother_perm_buy_med = (isp02c == 1) if !missing(isp02c) & isp02c != 6
gen mother_perm_buy_supply = (isp02d == 1) if !missing(isp02d) & isp02d != 6

//fertility questions 
gen ever_been_pregnant = (irh01 == 1) if !missing(irh01)
gen ever_given_birth = (irh02 == 1) if !missing(irh02)

gen ever_been_pregnant16to17 = ever_been_pregnant if iir01 >= 16 & iir01 <= 17
gen ever_been_pregnant18to21 = ever_been_pregnant if iir01 >= 18 & iir01 <= 21
gen ever_given_birth16to17 = ever_given_birth if iir01 >= 16 & iir01 <= 17
gen ever_given_birth18to21 = ever_given_birth if iir01 >= 18 & iir01 <= 21

//merge in pregnancy record 
merge 1:m iid using `outcomes_w1'
tab _merge  

//survey round indicator
gen survey_round = 0

tempfile mother_w1
save `mother_w1'

*Wave III
//code outcomes from pregnancy record file 
use "Wave III/Data/I_CH_fin", clear
duplicates report 

//coding pre-natal visits 
//do not include traditional birth visits (ich08g/ich10g/ich12g) or "other" (ich08v/ich10v/ich12v)
local prenatal ich08a ich08b ich08c ich08d ich08e ich08f ich08h ich08i ///
 			   ich10a ich10b ich10c ich10d ich10e ich10f ich10h ich10i ///
 			   ich12a ich12b ich12c ich12d ich12e ich12f ich12h ich12i 

//Recode missings as 0s
foreach x of var `prenatal' {
	replace `x' = 0 if `x' == . & ich07 != .
}

//calculate total number of pre-natal visits
egen pre_natal_visits = rowtotal(`prenatal'), missing 
replace pre_natal_visits = 20 if pre_natal_visits > 20 & pre_natal_visits < . //top-code unreasonable counts 


//coding dummy for >90 iron pills
gen iron_pills_dummy = 1 if ich21 == 4
replace iron_pills_dummy = 0 if ich21 < 4

gen iron_pills_categ = ich21
replace iron_pills_categ = 0 if ich20 == 3
label define ironcat 0 "Did not receive any" 1 "1-30 pills" 2 "31-60 pills" 3 "61-90 pills" 4 ">90 pills" 8 "Do Not Know, but did receive"
label values iron_pills_categ ironcat
replace iron_pills_categ = . if ich21 == 8


//coding assisted delivery
//generate dummy if doctor assisted in delivery 
gen dummy_A = strmatch(ich25, "*A*")
replace dummy_A = . if missing(ich25)
//dummy for midwife assisted delivery 
gen dummy_B = strmatch(ich25, "*B*")
replace dummy_B = . if missing(ich25)
//generate indicator for good assisted delivery 
gen good_assisted_delivery = (dummy_A == 1 | dummy_B == 1) if !missing(ich25)


//coding delivery at facility
gen delivery_facility = (ich24 < 7)
replace delivery_facility = . if missing(ich24)


//coding post-natal visits 
local postnatal ich32a ich32b ich32c ich32d ich32e ich32f ich32h ich32i ///
 				ich33a ich33b ich33c ich33d ich33e ich33f ich33h ich33i 
//Recode missings as 0s
foreach x of var `postnatal' {
	replace `x' = 0 if `x'== . & ich31 != .
}

//calculate total number of post-natal visits 
egen post_natal_visits = rowtotal(`postnatal'), missing 
replace post_natal_visits = 10 if post_natal_visits > 10 & post_natal_visits < . //top-code unreasonable counts


//birthweight missing indicator as well as actual weight
//missing indicator only for live births 
gen missing_birthweight = (missing(ich30)) if ich02 == 4
gen knows_birthweight = 1 - missing_birthweight if ich02 == 4
gen birthweight = ich30 if ich02 == 4

//coding miscarriage and miscarriage/stillbirth 
gen miscarriage = (ich02 == 2) if !missing(ich02) & ich02 != 1
gen miscarriage_or_stillborn = (ich02 == 2 | ich02 == 3) if !missing(ich02) & ich02 != 1

tempfile outcomes_w3
save `outcomes_w3', replace 

//merge to full woman 16 to 49 record 
use "Wave III/Data/i_fin", clear
duplicates report iid //2 exact duplicates 
duplicates drop 

//Code knowledge on breastfeeding and diarrhea (for women with kids under age 3 years)
gen q1 = 1 if ifks01_n >= 4 & ifks01_n < .
replace q1 = 0 if ifks01 == 8 | ifks01_n < 4

gen q2 = (ifks02 == 1)
gen q3 = (ifks03_n == 180)
gen q4 = (ifks04 == 1)
gen q5 = (ifks05 == 2)

egen q_sum = rowtotal(q1 q2 q3 q4 q5)
gen mother_knowledge = q_sum / 5

//indicator for pre-natal visits question correct 
rename q1 pre_natal_question_correct  

//composite breastfeeding knowledge indicator 
egen breastfeed_correct_num = rowtotal(q2 q3)
gen breastfeed_correct_pct = breastfeed_correct_num / 2
gen breastfeed_correct_all = (breastfeed_correct_pct == 1)

//composite diarrhea knowledge indicator
egen diarrhea_correct_num = rowtotal(q4 q5)
gen diarrhea_correct_pct = diarrhea_correct_num / 2
gen diarrhea_correct_all = (diarrhea_correct_pct == 1)

//age-restricted knowledge outcomes 
gen mother_knowledge_35 = mother_knowledge if iir01 < 35
gen breastfeed_correct_pct_35 = breastfeed_correct_pct if iir01 < 35
gen diarrhea_correct_pct_35 = diarrhea_correct_pct if iir01 < 35
gen pre_natal_question_correct_35 = pre_natal_question_correct if iir01 < 35

//empowerment variables 
gen mother_inv_education = regexm(isp01a, "A") if !missing(isp01a) & isp01a != "W" // do not consider "N/A"
gen mother_excl_education = (isp01a == "A") if !missing(isp01a) & isp01a != "W"

gen mother_inv_health = regexm(isp01b, "A") if !missing(isp01b) & isp01b != "W" // do not consider "N/A"
gen mother_excl_health = (isp01b == "A") if !missing(isp01b) & isp01b != "W"

gen mother_inv_discipline = regexm(isp01c, "A") if !missing(isp01c) & isp01c != "W" // do not consider "N/A"
gen mother_excl_discipline = (isp01c == "A") if !missing(isp01c) & isp01c != "W"

gen mother_inv_birth = regexm(isp01d, "A") if !missing(isp01d) & isp01d != "W" // do not consider "N/A"
gen mother_excl_birth = (isp01d == "A") if !missing(isp01d) & isp01d != "W"

//mother has to ask permission to buy items 
gen mother_perm_buy_veg = (isp02a == 1) if !missing(isp02a) & isp02a != 6
gen mother_perm_buy_clothing = (isp02b == 1) if !missing(isp02b) & isp02b != 6
gen mother_perm_buy_med = (isp02c == 1) if !missing(isp02c) & isp02c != 6
gen mother_perm_buy_supply = (isp02d == 1) if !missing(isp02d) & isp02d != 6

//fertility questions 
gen ever_been_pregnant = (irh01 == 1) if !missing(irh01)
gen ever_given_birth = (irh02 == 1) if !missing(irh02)

gen ever_been_pregnant16to17 = ever_been_pregnant if iir01 >= 16 & iir01 <= 17
gen ever_been_pregnant18to21 = ever_been_pregnant if iir01 >= 18 & iir01 <= 21
gen ever_given_birth16to17 = ever_given_birth if iir01 >= 16 & iir01 <= 17
gen ever_given_birth18to21 = ever_given_birth if iir01 >= 18 & iir01 <= 21


//merge in pregnancy record with outcome variables 
merge 1:m iid using `outcomes_w3'
tab _merge 

//create 9-digit (Wave I) iid
rename iid iid_w3
gen iid = substr(iid_w3, 1, 9) + substr(iid_w3, 14, 2)

//survey round indicator
gen survey_round = 1

tempfile mother_w3
save `mother_w3', replace 

*Wave IV
use "Wave IV/I_CH", clear 
duplicates report 

//destring variables for pre-natal visit numbers where necessary
replace ich10v = "0" if ich10v == "TT" // take care of 2 non-numeric values of ich10v
destring ich08v ich10v ich12v, replace

//coding pre-natal visits 
//do not include traditional birth visits (ich08g/ich10g/ich12g) or "other" (ich08v/ich10v/ich12v)
local prenatal ich08a ich08b ich08c ich08d ich08e ich08f ich08h ich08i ///
 			   ich10a ich10b ich10c ich10d ich10e ich10f ich10h ich10i ///
 			   ich12a ich12b ich12c ich12d ich12e ich12f ich12h ich12i 

//Recode missings as 0s
foreach x of var `prenatal' {
	replace `x' = 0 if `x' == . & ich07 != .
}

//calculate total number of pre-natal visits
egen pre_natal_visits = rowtotal(`prenatal'), missing 
replace pre_natal_visits = 20 if pre_natal_visits > 20 & pre_natal_visits < . //top-code unreasonable counts 

//coding dummy for >90 iron pills
gen iron_pills_dummy = 1 if ich21 == 4
replace iron_pills_dummy = 0 if ich21 < 4

gen iron_pills_categ = ich21
replace iron_pills_categ = 0 if ich20 == 3
label define ironcat 0 "Did not receive any" 1 "1-30 pills" 2 "31-60 pills" 3 "61-90 pills" 4 ">90 pills" 8 "Do Not Know, but did receive"
label values iron_pills_categ ironcat
replace iron_pills_categ = . if ich21 == 8

//coding assisted delivery
//generate dummy if doctor assisted in delivery 
gen dummy_A = strmatch(ich25, "*A*")
replace dummy_A = . if missing(ich25)
//dummy for midwife assisted delivery 
gen dummy_B = strmatch(ich25, "*B*")
replace dummy_B = . if missing(ich25)
//generate indicator for good assisted delivery 
gen good_assisted_delivery = (dummy_A == 1 | dummy_B == 1) if !missing(ich25)


//coding delivery at facility
gen delivery_facility = (ich24 < 7)
replace delivery_facility = . if missing(ich24)


//coding post-natal visits (note inclusion of ich32i and ich33i)
local postnatal ich32a ich32b ich32c ich32d ich32e ich32f ich32h ich32i ///
 				ich33a ich33b ich33c ich33d ich33e ich33f ich33h ich33i 
//Recode missings as 0s
foreach x of var `postnatal' {
	replace `x' = 0 if `x'== . & ich31 != .
}

//calculate total number of post-natal visits 
egen post_natal_visits = rowtotal(`postnatal'), missing 
replace post_natal_visits = 10 if post_natal_visits > 10 & post_natal_visits < . //top-code unreasonable counts


//birthweight missing indicator as well as actual weight
//missing indicator only for live births 
gen missing_birthweight = (missing(ich30)) if ich02 == 4
gen knows_birthweight = 1 - missing_birthweight if ich02 == 4
gen birthweight = ich30 if ich02 == 4

//coding miscarriage and miscarriage/stillbirth 
gen miscarriage = (ich02 == 2) if !missing(ich02) & ich02 != 1
gen miscarriage_or_stillborn = (ich02 == 2 | ich02 == 3) if !missing(ich02) & ich02 != 1


tempfile outcomes_w4
save `outcomes_w4', replace 

//merge to full woman 16 to 49 record 
use "Wave IV/I", clear
duplicates report iid 

//Code knowledge on breastfeeding and diarrhea (for women with kids under age 3 years)
gen q1 = 1 if ifks01_n >= 4 & ifks01_n < .
replace q1 = 0 if ifks01 == 8 | ifks01_n < 4

gen q2 = (ifks02 == 1)
gen q3 = (ifks03_n == 180)
gen q4 = (ifks04 == 1)
gen q5 = (ifks05 == 2)

egen q_sum = rowtotal(q1 q2 q3 q4 q5)
gen mother_knowledge = q_sum / 5

//indicator for pre-natal visits question correct 
rename q1 pre_natal_question_correct  

//composite breastfeeding knowledge indicator 
egen breastfeed_correct_num = rowtotal(q2 q3)
gen breastfeed_correct_pct = breastfeed_correct_num / 2
gen breastfeed_correct_all = (breastfeed_correct_pct == 1)

//composite diarrhea knowledge indicator
egen diarrhea_correct_num = rowtotal(q4 q5)
gen diarrhea_correct_pct = diarrhea_correct_num / 2
gen diarrhea_correct_all = (diarrhea_correct_pct == 1)

//age-restricted knowledge outcomes 
gen mother_knowledge_35 = mother_knowledge if iir01 < 35
gen breastfeed_correct_pct_35 = breastfeed_correct_pct if iir01 < 35
gen diarrhea_correct_pct_35 = diarrhea_correct_pct if iir01 < 35
gen pre_natal_question_correct_35 = pre_natal_question_correct if iir01 < 35

//empowerment variables 
gen mother_inv_education = regexm(isp01a, "A") if !missing(isp01a) & isp01a != "W" // do not consider "N/A"
gen mother_excl_education = (isp01a == "A") if !missing(isp01a) & isp01a != "W"
gen mother_excl_education_def2 = (isp01aa == 1) if !missing(isp01aa) & isp01aa != 96

gen mother_inv_health = regexm(isp01b, "A") if !missing(isp01b) & isp01b != "W" // do not consider "N/A"
gen mother_excl_health = (isp01b == "A") if !missing(isp01b) & isp01b != "W"
gen mother_excl_health_def2 = (isp01ab == 1) if !missing(isp01ab) & isp01ab != 96

gen mother_inv_discipline = regexm(isp01c, "A") if !missing(isp01c) & isp01c != "W" // do not consider "N/A"
gen mother_excl_discipline = (isp01c == "A") if !missing(isp01c) & isp01c != "W"
gen mother_excl_discipline_def2 = (isp01ac == 1) if !missing(isp01ac) & isp01ac != 96

gen mother_inv_birth = regexm(isp01d, "A") if !missing(isp01d) & isp01d != "W" // do not consider "N/A"
gen mother_excl_birth = (isp01d == "A") if !missing(isp01d) & isp01d != "W"
gen mother_excl_birth_def2 = (isp01ad == 1) if !missing(isp01ad) & isp01ad != 96

//mother has to ask permission to buy items 
gen mother_perm_buy_veg = (isp02a == 1) if !missing(isp02a) & isp02a != 6
gen mother_perm_buy_clothing = (isp02b == 1) if !missing(isp02b) & isp02b != 6
gen mother_perm_buy_med = (isp02c == 1) if !missing(isp02c) & isp02c != 6
gen mother_perm_buy_supply = (isp02d == 1) if !missing(isp02d) & isp02d != 6

//fertility questions 
gen ever_been_pregnant = (irh01 == 1) if !missing(irh01)
gen ever_given_birth = (irh02 == 1) if !missing(irh02)

gen ever_been_pregnant16to17 = ever_been_pregnant if iir01 >= 16 & iir01 <= 17
gen ever_been_pregnant18to21 = ever_been_pregnant if iir01 >= 18 & iir01 <= 21
gen ever_given_birth16to17 = ever_given_birth if iir01 >= 16 & iir01 <= 17
gen ever_given_birth18to21 = ever_given_birth if iir01 >= 18 & iir01 <= 21

//merge in pregnancy record with outcome variables 
merge 1:m iid using `outcomes_w4'
tab _merge  

//generate 9-digit aid variable
rename iid iid_w4
gen iid = substr(iid_w4, 1, 7) + substr(iid_w4, 10, 2) + substr(iid_w4, 18, 2)

//survey round indicator
gen survey_round = 2

tempfile mother_w4
save `mother_w4', replace 


*** 4. APPEND ALL WAVES ***
//Use Wave I merged file 
use `mother_w1', clear

//Append Wave III and IV observations 
append using `mother_w3' `mother_w4', force
sort iid survey_round 

tempfile mother_allwaves
save `mother_allwaves', replace


*** 5. CODING PKH RANDOMIZATION/TREATMENT STATUS ***
//Open PKH/Kec data from Ekki
use "Wave I/Data/pkhben2013_use.dta", clear

//kep only ea, randomization, PKH status, province code
keep ea L07 K09 K13 prov09 

//keep one observation per ea (kecamatan)
duplicates drop ea, force

//drop extra 90 kecs in order to restrict to original 360 kecamatans
drop if missing(L07)

tempfile ea_treatment_list
save `ea_treatment_list'

//use merged all-wave household data
use `mother_allwaves', clear

//generate ea string
gen ea = substr(iid, 1, 3)

//merge with kecamatan treatment list 
merge m:1 ea using `ea_treatment_list', generate(merge_treat)
tab merge_treat
//only keep observations from original 360 kecamatans (L07 is non-missing)
keep if merge_treat == 3
drop merge_treat
sort iid survey_round


*** 6. MERGING HOUSEHOLD-LEVEL COVARIATES ***
//create rid variable to match children with households from HH file
gen rid = substr(iid, 1, 3) + "1" + substr(iid, 5, 5)

tempfile mother_allwaves_coded
save `mother_allwaves_coded', replace 

//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only relevant household covariates 
keep rid* survey_round received* pkh* L07 hh_head_education hh_head_agr ///
	 hh_head_serv hh_educ* roof_type* wall_type* floor_type* clean_water own_latrine ///
	 square_latrine own_septic_tank electricity_PLN logpcexp* hhsize_ln *baseline_nm *miss ///
	 kabupaten kecamatan province lack* low* num_*_pc_bl total_*_pc_bl non_pkh_hh_cct_kec

tempfile hh_covariates
save `hh_covariates', replace

//merge 
use `mother_allwaves_coded', clear 
//generate rid_merge variable to merge to correct household (accounts for HH splits after baseline)
gen rid_merge = substr(iid, 1, 3) + "1" + substr(iid,5,5) if survey_round == 0
replace rid_merge = substr(iid_w3, 1, 3) + "1" + substr(iid_w3,5,9) if survey_round == 1
replace rid_merge = substr(iid_w4, 1, 3) + "1" + substr(iid_w4,5,13) if survey_round == 2 


merge m:1 rid_merge survey_round using `hh_covariates', generate(_merge_hh)
tab _merge_hh
drop if _merge_hh == 2
drop _merge_hh


//save
compress
cd "`PKH'data/coded"
save "marriedwoman16to49_allwaves_master", replace


