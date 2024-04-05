

version 14.1 		
set more off		
clear all			
pause on 			


//Prompt for directory if not filled in above
if "$PKH" == "" {
	display "Enter the directory where the PKH files can be located: " _request(PKH)
	* add a \ if they forgot one
	quietly if substr("$PKH",length("PKH"),1) != "/" & substr("$PKH",length("$PKH"),1) != "\" {
		global PKH = "$PKH\"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local latex = "`output'/latex"
local TARGET2 = "$TARGET2"


//Set working directory:
cd "`pkhdata'"


*** 3. CODE VILLAGE-LEVEL HEALTH SERVICES ***
*Wave III
use "Wave III/Data/L_FKS2TYPE_fin.dta", clear

//replace "TT" and "96" with missing
replace lfks06 = "" if lfks06 == "TT" | lfks06 == "96"
replace lfks07 = "" if lfks07 == "TT" | lfks07 == "96"
destring lfks06 lfks07, replace

//reshape for 1 obs per village
reshape wide lfks06 lfks07, i(lid) j(lfks2type)
//rename relevant variables
forv i = 1/7 {
	local category: word `i' of "maledoc" "femaledoc" "totaldoc" "nurse" "villmidwife" "midwife" "tradbirth"
	rename lfks06`i' num_`category'_prac
	rename lfks07`i' num_`category'_live
}
//generate category for total number of midwives 
gen num_totalmidwife_prac = num_villmidwife_prac + num_midwife_prac 
gen num_totalmidwife_live = num_villmidwife_live + num_midwife_live 

//survey round indicator
gen survey_round = 1

tempfile village_health_w3
save `village_health_w3', replace 



*Wave IV
use "Wave IV/L_FKS2TYPE.dta", clear 

//replace .y with normal missing
replace lfks06 = . if lfks06 == .y
replace lfks07 = . if lfks07 == .y

//generate unique village identifier incorporating split code 
gen lid_with_split = lid + lsplit 

//reshape for 1 obs per village 
reshape wide lfks06 lfks07, i(lid_with_split) j(lfks2type)
//rename relevant variables
forv i = 1/7 {
	local category: word `i' of "maledoc" "femaledoc" "totaldoc" "nurse" "villmidwife" "midwife" "tradbirth"
	rename lfks06`i' num_`category'_prac
	rename lfks07`i' num_`category'_live
}
//generate category for total number of midwives 
gen num_totalmidwife_prac = num_villmidwife_prac + num_midwife_prac 
gen num_totalmidwife_live = num_villmidwife_live + num_midwife_live 

//survey round indicator
gen survey_round = 2

tempfile village_health_w4
save `village_health_w4', replace 



*Wave I (baseline)
use "Wave I/Data/L_FKS2TYPE_fin.dta", clear 

//replace .y and .z with normal missing
replace lfks06 = . if lfks06 == .y | lfks06 == .z
replace lfks07 = . if lfks07 == .y | lfks07 == .z

//reshape for 1 obs per village 
reshape wide lfks06 lfks07, i(lid) j(lfks2type)
//rename relevant variables
forv i = 1/7 {
	local category: word `i' of "maledoc" "femaledoc" "totaldoc" "nurse" "villmidwife" "midwife" "tradbirth"
	rename lfks06`i' num_`category'_prac
	rename lfks07`i' num_`category'_live
}
//generate category for total number of midwives 
gen num_totalmidwife_prac = num_villmidwife_prac + num_midwife_prac 
gen num_totalmidwife_live = num_villmidwife_live + num_midwife_live 

//survey round indicator
gen survey_round = 0

append using `village_health_w3' `village_health_w4'
sort lid survey_round

//label those observations that match to baseline 
bysort lid (survey_round): gen baseline_match = (survey_round[1] == 0)
//generate variables for baseline value of each category 
forv i = 1/8 {
	local category: word `i' of "maledoc" "femaledoc" "totaldoc" "nurse" "villmidwife" "midwife" "tradbirth" "totalmidwife"
	bysort lid (survey_round): gen num_`category'_prac_bl = num_`category'_prac[1] if baseline_match == 1 
	bysort lid (survey_round): gen num_`category'_live_bl = num_`category'_live[1] if baseline_match == 1 
	
	//generate non-missing and missing baseline variables 
	gen num_`category'_prac_bl_nm = num_`category'_prac_bl 
	gen num_`category'_prac_bl_miss = (missing(num_`category'_prac_bl))
	replace num_`category'_prac_bl_nm = 0 if missing(num_`category'_prac_bl_nm)

	gen num_`category'_live_bl_nm = num_`category'_live_bl 
	gen num_`category'_live_bl_miss = (missing(num_`category'_live_bl))
	replace num_`category'_live_bl_nm = 0 if missing(num_`category'_live_bl_nm)
}


//lid for merging with school characteristics
gen lid_merge = lid
replace lid_merge = lid_with_split if survey_round == 2

tempfile village_health_allwave
save `village_health_allwave', replace 




*** 4. CODE VILLAGE-LEVEL SCHOOLING FACILITIES ***
*Wave III
use "Wave III/Data/L_FPDTYPE_fin.dta", clear

//replace "TT" with missing
replace lfpd02 = "" if lfpd02 == "TT"
replace lfpd03 = "" if lfpd03 == "TT"
destring lfpd02 lfpd03, replace

//generate variable of interest: total number of schools (don't need individual types)
gen num_school = lfpd02 + lfpd03
drop lfpd02 lfpd03

//reshape for 1 obs per village
reshape wide num_school, i(lid) j(lfpdtype)

//generate outcomes of interest (primary and secondary schools)
gen total_primary = num_school2 + num_school3 + num_school4
gen total_secondary = num_school5 + num_school6 + num_school7 + num_school8 + num_school9

//survey round indicator 
gen survey_round = 1

tempfile village_school_w3
save `village_school_w3', replace 



*Wave IV
use "Wave IV/L_FPDTYPE.dta", clear
//replace .y with normal missing
replace lfpd02 = . if lfpd02 == .y
replace lfpd03 = . if lfpd03 == .y

//generate unique village identifier incorporating split code 
gen lid_with_split = lid + lsplit 

//generate variable of interest: total number of schools (don't need individual types)
gen num_school = lfpd02 + lfpd03
drop lfpd02 lfpd03

//reshape for 1 obs per village
reshape wide num_school, i(lid_with_split) j(lfpdtype)

//generate outcomes of interest (primary and secondary schools)
gen total_primary = num_school2 + num_school3 + num_school4
gen total_secondary = num_school5 + num_school6 + num_school7 + num_school8 + num_school9

//survey round indicator 
gen survey_round = 2

tempfile village_school_w4
save `village_school_w4', replace 



*Wave I (baseline)
use "Wave I/Data/L_FPDTYPE_fin.dta", clear
//replace .y with normal missing
replace lfpd02 = . if lfpd02 == .y | lfpd02 == .z
replace lfpd03 = . if lfpd03 == .y | lfpd03 == .z

//generate variable of interest: total number of schools (don't need individual types)
gen num_school = lfpd02 + lfpd03
drop lfpd02 lfpd03

//reshape for 1 obs per village
reshape wide num_school, i(lid) j(lfpdtype)

//generate outcomes of interest (primary and secondary schools)
gen total_primary = num_school2 + num_school3 + num_school4
gen total_secondary = num_school5 + num_school6 + num_school7 + num_school8 + num_school9

//survey round indicator 
gen survey_round = 0

append using `village_school_w3' `village_school_w4'
sort lid survey_round


//label those observations that match to baseline 
bysort lid (survey_round): gen baseline_match = (survey_round[1] == 0)
//generate variables for baseline value of each category 
forv i = 1/2 {
	local category: word `i' of "total_primary" "total_secondary"
	bysort lid (survey_round): gen `category'_bl = `category'[1] if baseline_match == 1 
	
	//generate non-missing and missing baseline variables 
	gen `category'_bl_nm = `category'_bl 
	gen `category'_bl_miss = (missing(`category'_bl))
	replace `category'_bl_nm = 0 if missing(`category'_bl_nm)
}

tempfile village_school_allwave
save `village_school_allwave', replace 


//lid for merging with health characteristics
gen lid_merge = lid
replace lid_merge = lid_with_split if survey_round == 2



//merge with health characteristics 
merge 1:1 lid_merge survey_round using `village_health_allwave'
assert _merge == 3
drop _merge 


tempfile village_health_school_all
save `village_health_school_all', replace 


*** VILLAGE POVERTY ERADICATION PROGRAMS ***
*Wave III 
use "Wave III/Data/L_PAP_fin.dta", clear 

//for robustness, look for PKH lines in 3 different ways
//first, just look at lines where PAP03 code == 8 as indicated in questionnaire  
gen pkh_def1 = (lpap03_cd == 8)
//definition 2: lines where PAP03 code == 8 AND name has something to do with PKH 
gen pkh_def2 = (pkh_def1 == 1 & (regexm(lpap03, "PKH") | regexm(lpap03, "HARAPAN") | regexm(lpap03, "PHK") | regexm(lpap03, "YKH")))
//definition 3: def 2 + add lines where PAP03 code != 8 but name has something to do with PKH 
gen pkh_def3 = pkh_def2 
replace pkh_def3 = 1 if (regexm(lpap03, "PKH") | regexm(lpap03, "KELUARGA HARAPAN") | regexm(lpap03, "PHK") | regexm(lpap03, "YKH")) & lpap03_cd != 8

//total budget for PKH 
gen pkh_budget_def1 = lpap05_n if pkh_def1 == 1
gen pkh_budget_def2 = lpap05_n if pkh_def2 == 1
gen pkh_budget_def3 = lpap05_n if pkh_def3 == 1
gen pkh_budget_dontknow_def1 = (lpap05 == 8) if pkh_def1 == 1
gen pkh_budget_dontknow_def2 = (lpap05 == 8) if pkh_def2 == 1
gen pkh_budget_dontknow_def3 = (lpap05 == 8) if pkh_def3 == 1

//number of households 
gen num_pkh_hh_def1 = lpap05a_n if pkh_def1 == 1
gen num_pkh_hh_def2 = lpap05a_n if pkh_def2 == 1
gen num_pkh_hh_def3 = lpap05a_n if pkh_def3 == 1
gen num_pkh_hh_dontknow_def1 = (lpap05a == 8) if pkh_def1 == 1
gen num_pkh_hh_dontknow_def2 = (lpap05a == 8) if pkh_def2 == 1
gen num_pkh_hh_dontknow_def3 = (lpap05a == 8) if pkh_def3 == 1


//collapse to 1 observation per village and sum total budget for assistance programs 
collapse (sum) village_tot_assistance_budget=lpap05_n pkh_budget_def1 pkh_budget_def2 pkh_budget_def3 num_pkh_hh_def1 num_pkh_hh_def2 num_pkh_hh_def3 ///
				(max) any_pkh_def1=pkh_def1 any_pkh_def2=pkh_def2 any_pkh_def3=pkh_def3 pkh_budget_dontknow_def1 pkh_budget_dontknow_def2 pkh_budget_dontknow_def3 ///
				num_pkh_hh_dontknow_def1 num_pkh_hh_dontknow_def2 num_pkh_hh_dontknow_def3, by(lid)

//tempfile 
gen survey_round = 1
tempfile programs_w3
save `programs_w3', replace 


*Wave IV
use "Wave IV/L_PAP.dta", clear 


//first, just look at lines where PAP03 code == 8 as indicated in questionnaire  
gen pkh_def1 = (lpap03_cd == 8)
//definition 2: lines where PAP03 code == 8 AND name has something to do with PKH 
gen pkh_def2 = (pkh_def1 == 1 & (regexm(lpap03, "PKH") | regexm(lpap03, "HARAPAN") | regexm(lpap03, "PHK") | regexm(lpap03, "PKG")))
//definition 3: def 2 + add lines where PAP03 code != 8 but name has something to do with PKH 
gen pkh_def3 = pkh_def2 
replace pkh_def3 = 1 if (regexm(lpap03, "PKH") | regexm(lpap03, "KELUARGA HARAPAN") | regexm(lpap03, "PHK") | regexm(lpap03, "YKH")) & lpap03_cd != 8

//total budget for PKH 
gen pkh_budget_def1 = lpap05_n if pkh_def1 == 1
gen pkh_budget_def2 = lpap05_n if pkh_def2 == 1
gen pkh_budget_def3 = lpap05_n if pkh_def3 == 1
gen pkh_budget_dontknow_def1 = (lpap05 == 8) if pkh_def1 == 1
gen pkh_budget_dontknow_def2 = (lpap05 == 8) if pkh_def2 == 1
gen pkh_budget_dontknow_def3 = (lpap05 == 8) if pkh_def3 == 1

//number of households 
gen num_pkh_hh_def1 = lpap05a_n if pkh_def1 == 1
gen num_pkh_hh_def2 = lpap05a_n if pkh_def2 == 1
gen num_pkh_hh_def3 = lpap05a_n if pkh_def3 == 1
gen num_pkh_hh_dontknow_def1 = (lpap05a == 8) if pkh_def1 == 1
gen num_pkh_hh_dontknow_def2 = (lpap05a == 8) if pkh_def2 == 1
gen num_pkh_hh_dontknow_def3 = (lpap05a == 8) if pkh_def3 == 1

//collapse to 1 observation per village and sum total budget for assistance programs 
collapse (sum) village_tot_assistance_budget=lpap05_n pkh_budget_def1 pkh_budget_def2 pkh_budget_def3 num_pkh_hh_def1 num_pkh_hh_def2 num_pkh_hh_def3 ///
				(max) any_pkh_def1=pkh_def1 any_pkh_def2=pkh_def2 any_pkh_def3=pkh_def3 pkh_budget_dontknow_def1 pkh_budget_dontknow_def2 pkh_budget_dontknow_def3 ///
				num_pkh_hh_dontknow_def1 num_pkh_hh_dontknow_def2 num_pkh_hh_dontknow_def3, by(lid lsplit)

//generate unique village identifier incorporating split code 
gen lid_with_split = lid + lsplit 

//tempfile 
gen survey_round = 2

*Wave I
*[PAP MODULE NOT INCLUDED IN WAVE I]


//append and save 
append using `programs_w3'
sort lid lid_with_split survey_round

//lid for merge 
gen lid_merge = lid
replace lid_merge = lid_with_split if survey_round == 2

tempfile programs_all
save `programs_all', replace 


*** MAIN VILLAGE OUTCOMES ***
*Wave III
use "Wave III/Data/L_fin.dta", clear 
duplicates drop 
//take care of one anomalous duplicate
drop if lid == "0825001" & lfpd01_a_cd == 17


//for each problem, create an indicator for whether village head identified it as most serious problem, and separately for top 3 serious problems 
*health services
gen lack_health_fac_mostserious = (lfks01_a_cd == 1)
gen lack_health_fac_top3 = (lfks01_a_cd == 1 | lfks01_b_cd == 1 | lfks01_c_cd == 1)

gen lack_med_equip_mostserious = (lfks01_a_cd == 2)
gen lack_med_equip_top3 = (lfks01_a_cd == 2 | lfks01_b_cd == 2 | lfks01_c_cd == 2)

gen lack_hlthworkers_mostserious = (lfks01_a_cd == 3)
gen lack_hlthworkers_top3 = (lfks01_a_cd == 3 | lfks01_b_cd == 3 | lfks01_c_cd == 3)

gen low_hlth_aware_mostserious = (lfks01_a_cd == 9)
gen low_hlth_aware_top3 = (lfks01_a_cd == 9 | lfks01_b_cd == 9 | lfks01_c_cd == 9)

*education (SD)
gen lack_sd_fac_mostserious = (lfpd00_a_cd == 1)
gen lack_sd_fac_top3 = (lfpd00_a_cd == 1 | lfpd00_b_cd == 1 | lfpd00_c_cd == 1) 

gen lack_sd_infr_mostserious = (lfpd00_a_cd == 2)
gen lack_sd_infr_top3 = (lfpd00_a_cd == 2 | lfpd00_b_cd == 2 | lfpd00_c_cd == 2)

gen lack_sd_teachers_mostserious = (lfpd00_a_cd == 3)
gen lack_sd_teachers_top3 = (lfpd00_a_cd == 3 | lfpd00_b_cd == 3 | lfpd00_c_cd == 3)

gen lack_sd_aware_mostserious = (lfpd00_a_cd == 6)
gen lack_sd_aware_top3 = (lfpd00_a_cd == 6 | lfpd00_b_cd == 6 | lfpd00_c_cd == 6)

*education (SMP)
gen lack_smp_fac_mostserious = (lfpd01_a_cd == 1)
gen lack_smp_fac_top3 = (lfpd01_a_cd == 1 | lfpd01_b_cd == 1 | lfpd01_c_cd == 1) 

gen lack_smp_infr_mostserious = (lfpd01_a_cd == 2)
gen lack_smp_infr_top3 = (lfpd01_a_cd == 2 | lfpd01_b_cd == 2 | lfpd01_c_cd == 2)

gen lack_smp_teachers_mostserious = (lfpd01_a_cd == 3)
gen lack_smp_teachers_top3 = (lfpd01_a_cd == 3 | lfpd01_b_cd == 3 | lfpd01_c_cd == 3)

gen lack_smp_aware_mostserious = (lfpd01_a_cd == 6)
gen lack_smp_aware_top3 = (lfpd01_a_cd == 6 | lfpd01_b_cd == 6 | lfpd01_c_cd == 6)


//survey round indicator 
gen survey_round = 1

tempfile mainvillage_w3
save `mainvillage_w3', replace 



*Wave IV
use "Wave IV/L.dta", clear 

//for each problem, create an indicator for whether village head identified it as most serious problem, and separately for top 3 serious problems 
*health services
gen lack_health_fac_mostserious = (lfks01_a_cd == 1)
gen lack_health_fac_top3 = (lfks01_a_cd == 1 | lfks01_b_cd == 1 | lfks01_c_cd == 1)

gen lack_med_equip_mostserious = (lfks01_a_cd == 2)
gen lack_med_equip_top3 = (lfks01_a_cd == 2 | lfks01_b_cd == 2 | lfks01_c_cd == 2)

gen lack_hlthworkers_mostserious = (lfks01_a_cd == 3)
gen lack_hlthworkers_top3 = (lfks01_a_cd == 3 | lfks01_b_cd == 3 | lfks01_c_cd == 3)

gen low_hlth_aware_mostserious = (lfks01_a_cd == 9)
gen low_hlth_aware_top3 = (lfks01_a_cd == 9 | lfks01_b_cd == 9 | lfks01_c_cd == 9)

*education (SD)
gen lack_sd_fac_mostserious = (lfpd00_a_cd == 1)
gen lack_sd_fac_top3 = (lfpd00_a_cd == 1 | lfpd00_b_cd == 1 | lfpd00_c_cd == 1) 

gen lack_sd_infr_mostserious = (lfpd00_a_cd == 2)
gen lack_sd_infr_top3 = (lfpd00_a_cd == 2 | lfpd00_b_cd == 2 | lfpd00_c_cd == 2)

gen lack_sd_teachers_mostserious = (lfpd00_a_cd == 3)
gen lack_sd_teachers_top3 = (lfpd00_a_cd == 3 | lfpd00_b_cd == 3 | lfpd00_c_cd == 3)

gen lack_sd_aware_mostserious = (lfpd00_a_cd == 6)
gen lack_sd_aware_top3 = (lfpd00_a_cd == 6 | lfpd00_b_cd == 6 | lfpd00_c_cd == 6)

*education (SMP)
gen lack_smp_fac_mostserious = (lfpd01_a_cd == 1)
gen lack_smp_fac_top3 = (lfpd01_a_cd == 1 | lfpd01_b_cd == 1 | lfpd01_c_cd == 1) 

gen lack_smp_infr_mostserious = (lfpd01_a_cd == 2)
gen lack_smp_infr_top3 = (lfpd01_a_cd == 2 | lfpd01_b_cd == 2 | lfpd01_c_cd == 2)

gen lack_smp_teachers_mostserious = (lfpd01_a_cd == 3)
gen lack_smp_teachers_top3 = (lfpd01_a_cd == 3 | lfpd01_b_cd == 3 | lfpd01_c_cd == 3)

gen lack_smp_aware_mostserious = (lfpd01_a_cd == 6)
gen lack_smp_aware_top3 = (lfpd01_a_cd == 6 | lfpd01_b_cd == 6 | lfpd01_c_cd == 6)

//generate unique village identifier incorporating split code 
gen lid_with_split = lid + lsplit 

//survey round indicator 
gen survey_round = 2

tempfile mainvillage_w4
save `mainvillage_w4', replace 



*Wave I
use "Wave I/Data/L_fin.dta", clear 

//Wave I doesn't have numerically coded issue variables, so we look at the lettered categories. Unfortunately, 
//the letters do not indicate ranking of seriousness of issues
*health services
gen lack_health_fac_top3 = (regexm(lfks01, "A")) 
gen lack_med_equip_top3 = (regexm(lfks01, "B"))
gen lack_hlthworkers_top3 = (regexm(lfks01, "C"))
gen low_hlth_aware_top3 = (regexm(lfks01, "I"))

*education (again unfortunately, this survey round does not include info specifically about SD/SMP)
gen lack_sch_fac_top3 = (regexm(lfpd01, "A"))
gen lack_sch_infr_top3 = (regexm(lfpd01, "B"))
gen lack_sch_teachers_top3 = (regexm(lfpd01, "C"))
gen lack_sch_aware_top3 = (regexm(lfpd01, "F"))


//survey round indicator 
gen survey_round = 0


//append other waves 
append using `mainvillage_w3' `mainvillage_w4', force 
sort lid survey_round


//label those observations that match to baseline 
bysort lid (survey_round): gen baseline_match_main = (survey_round[1] == 0)
//generate variables for baseline value of each category 
forv i = 1/8 {
	local category: word `i' of "lack_health_fac_top3" "lack_med_equip_top3" "lack_hlthworkers_top3" "low_hlth_aware_top3" ///
								"lack_sch_fac_top3" "lack_sch_infr_top3" "lack_sch_teachers_top3" "lack_sch_aware_top3"
	bysort lid (survey_round): gen `category'_bl = `category'[1] if baseline_match_main == 1 
	
	//generate non-missing and missing baseline variables 
	gen `category'_bl_nm = `category'_bl 
	gen `category'_bl_miss = (missing(`category'_bl))
	replace `category'_bl_nm = 0 if missing(`category'_bl_nm)
}


//lid for merge 
gen lid_merge = lid
replace lid_merge = lid_with_split if survey_round == 2


//merge with other outcomes 
merge 1:1 lid_merge survey_round using `programs_all'
assert _merge == 3 | survey_round == 0 //don't have variables on programs at baseline 
drop _merge 

merge 1:1 lid_merge survey_round using `village_health_school_all'
assert _merge == 3
drop _merge 


//code % of households with PKH 
gen pct_pkh_hh_def1 = num_pkh_hh_def1 / lid02
gen pct_pkh_hh_def2 = num_pkh_hh_def2 / lid02
gen pct_pkh_hh_def3 = num_pkh_hh_def3 / lid02

//code per-household and per-capita budget for programs 
gen pkh_pct_of_total_budget_1 = pkh_budget_def1 / village_tot_assistance_budget
gen pkh_pct_of_total_budget_2 = pkh_budget_def2 / village_tot_assistance_budget
gen pkh_pct_of_total_budget_3 = pkh_budget_def3 / village_tot_assistance_budget


//generate per-capita and per-household versions of supply variables 
foreach x of varlist num_totaldoc_live num_totaldoc_prac num_nurse_live num_nurse_prac num_totalmidwife_live num_totalmidwife_prac num_tradbirth_live num_tradbirth_prac total_primary total_secondary {
	gen `x'_pc = `x' / lid01 
	gen `x'_phh = `x' / lid02

	gen `x'_lnpc = ln((`x' / lid01) + 1)
	gen `x'_lnphh = ln((`x' / lid02) + 1)

	//baseline values
	bysort lid (survey_round): gen `x'_pc_bl = `x'_pc[1] if baseline_match == 1
	bysort lid (survey_round): gen `x'_phh_bl = `x'_phh[1] if baseline_match == 1
	gen `x'_pc_bl_nm = `x'_pc_bl 
	gen `x'_pc_bl_miss = (missing(`x'_pc_bl))
	replace `x'_pc_bl_nm = 0 if missing(`x'_pc_bl_nm)

	gen `x'_ph_bl_nm = `x'_phh_bl 
	gen `x'_ph_bl_miss = (missing(`x'_phh_bl))
	replace `x'_ph_bl_nm = 0 if missing(`x'_ph_bl_nm)
}


//generate ea variable for future merges 
gen ea = substr(lid, 1, 3)



cap drop llk01_* llk02_* llk03_*
sort lid survey_round 
tempfile village_outcomes_all
save `village_outcomes_all', replace 


*** 5. MERGE IN KECAMATAN AND KABUPATEN VARIABLES ***
cd "`pkhdata'"
//start with baseline and subsequent treatment status of kecamatan
use "Wave I/Data/pkhben2013_use.dta", clear 
//keep one observation per kecamatan
bysort ea: keep if _n == 1
//keep only treatment kecs 
drop if missing(L07)
//keep only necessary variables 
keep ea L07 K09 K13

tempfile kec_treatments
save `kec_treatments', replace 

//move on to kec/kab/province names (village-level dataset)
use "Wave I/Data/kecnames_ben.dta"

//merge with village characteristics 
merge 1:m lid using `village_outcomes_all'
assert _merge != 1
drop _merge

//carryforward kab/provinces to villages missing from baseline
cap drop llk03_nm llk03_cd  
bysort ea (lid): carryforward llk*, replace 
 
//rename variables 
rename llk01_cd province 
rename llk02_cd kabupaten 

//merge in treatment status 
merge m:1 ea using `kec_treatments'
keep if _merge == 3
drop _merge 
assert !missing(L07)

//generate numerical kecamatan code 
destring ea, generate(kecamatan)

*** 6. SAVE ***
sort lid survey_round lsplit 

compress
cd "`PKH'data/coded"
save "village_outcomes_master", replace



