*** 1. INTRODUCTION ***
/*
Description: Calculates Romano-Wolf p-values for Table 3
Uses: child6to15_allwaves_master	
Creates: rwolf_pvalues_table3.xls
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
		global PKH = "$PKH\"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local TARGET2 = "$TARGET2"


//set ado path for rwolf_pkh
adopath + "`PKH'ado"


*** 2. TABLE 3 (EDUCATION OUTCOMES) ***
//specify file for putexcel
putexcel set rwolf_pvalues_table3.xls, sheet(table3) replace

//set local with number of outcomes 
local panela_count = 2
local panelb_count = 3
local panelc_count = 3
local num_outcomes = `panela_count' + `panelb_count' + `panelc_count'
//set outcomes by panel HERE
local outcomes_a enroll_7to15 age7to15_85_twoweeks
local outcomes_b enroll_age7to12 enroll_age7to12_SD age7to12_85_twoweeks
local outcomes_c enroll_age13to15 enroll_age13to15_SMP age13to15_85_twoweeks
//declare matrix for storing p-values
matrix panel_a = J(`panela_count', 2, .)
matrix panel_b = J(`panelb_count', 2, .)
matrix panel_c = J(`panelc_count', 2, .)


**Data for all panels**
cd "`PKH'data/coded"
use "child6to15_allwaves_master.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics (INCLUDES LOG PCE)
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm hhsize_ln_baseline_nm logpcexp_baseline_nm *miss

//generate interaction variables
foreach x of varlist `baseline_controls' {
	generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)


//subset for faster runtime, keeping only necessary variables and observations
drop if missing(L07) | missing(kecamatan)
keep  enroll_7to15 age7to15_85_twoweeks enroll_age7to12 enroll_age7to12_SD age7to12_85_twoweeks ///
	 SMP_transition_ever_7to15 enroll_age13to15 enroll_age13to15_SMP age13to15_85_twoweeks ///
	`baseline_controls' `interactions' kabu* kecamatan survey_round pkh_by_this_wave L07


***PANEL A: ENROLLMENT FOR AGES 7-15***
matrix rownames panel_a = `outcomes_a'

//change to output directory and run RW 
cd "`output'"


//Column 1 (2-year, Lottery Only)
//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

rwolf_pkh `outcomes_a', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(30293812) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panela_count' {
	local rowname: word `i' of `outcomes_a'
	matrix panel_a[`i', 1] = e(rw_`rowname')
}


//Column 2 (6-year, Lottery Only)
restore
preserve
keep if survey_round == 2
rwolf_pkh `outcomes_a', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(20192186) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panela_count' {
	local rowname: word `i' of `outcomes_a'
	matrix panel_a[`i', 2] = e(rw_`rowname')
}


restore


***PANEL B: OUTCOMES FOR AGES 7-12***
matrix rownames panel_b = `outcomes_b'

//change to output directory and run RW 
cd "`output'"

//column 1
//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

//Column 1 (2-year, Lottery Only)
rwolf_pkh `outcomes_b', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(54328782) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelb_count' {
	local rowname: word `i' of `outcomes_b'
	matrix panel_b[`i', 1] = e(rw_`rowname')
}

//Column 2 (6-year, Lottery Only)
restore
preserve
keep if survey_round == 2
rwolf_pkh `outcomes_b', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(32065320) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelb_count' {
	local rowname: word `i' of `outcomes_b'
	matrix panel_b[`i', 2] = e(rw_`rowname')
}


restore



***PANEL C: OUTCOMES FOR AGES 13-15***
matrix rownames panel_c = `outcomes_c'

//change to output directory and run RW 
cd "`output'"

//column 1
//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

//Column 1 (2-year, Lottery Only)
rwolf_pkh `outcomes_c', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(15978952) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelc_count' {
	local rowname: word `i' of `outcomes_c'
	matrix panel_c[`i', 1] = e(rw_`rowname')
}


//Column 2 (6-year, Lottery Only)
restore
preserve
keep if survey_round == 2
rwolf_pkh `outcomes_c', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(97895102) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelc_count' {
	local rowname: word `i' of `outcomes_c'
	matrix panel_c[`i', 2] = e(rw_`rowname')
}

restore





***Export appended matrix***
matrix pvals = panel_a \ panel_b \ panel_c
putexcel A1 = matrix(pvals), names nformat(number_d3)


