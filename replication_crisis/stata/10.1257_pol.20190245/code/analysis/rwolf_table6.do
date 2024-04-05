*** 1. INTRODUCTION ***
/*
Description: Calculates Romano-Wolf p-values for Table 6
Uses: 	child6to15_allwaves_master.dta; child6to15_fertility_marriage_outcomes.dta
Creates: rwolf_pvalues_table6.xls
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


*** TABLE 6 (MEDIUM-RUN OUTCOMES) ***
//specify file for putexcel
putexcel set rwolf_pvalues_table6.xls, sheet(table6) replace

//set local with number of outcomes for each panel 
local panela_count_c1 = 2
local panela_count_c2 = 3
local panelb_count_c1 = 2
local panelb_count_c2 = 4
local panelc_count_c1 = 1
local panelc_count_c2 = 2
//declare matrices for storing p-values
matrix panel_a = J(`panela_count_c2', 2, .)
matrix panel_b = J(`panelb_count_c2', 2, .)
matrix panel_c = J(`panelc_count_c2', 2, .)









***
*PANEL A
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
keep  enroll_any_15to17 enroll_SMA_15to17 wageonly_16to17 wageonly_20hrs_16to17 ///
	completed_SMA_18to21 wageonly_18to21 wageonly_20hrs_18to21 ///
	`baseline_controls' `interactions' kabu* kecamatan survey_round pkh_by_this_wave L07


//local with outcome variables
local outcomes_c1 enroll_any_15to17 enroll_SMA_15to17
local outcomes_c2 enroll_any_15to17 enroll_SMA_15to17 completed_SMA_18to21
matrix rownames panel_a = `outcomes_c2'


***change to output directory and run RW 
cd "`output'"

//Column 1 (2-year, Lottery Only)
//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

rwolf_pkh `outcomes_c1', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(20301948) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panela_count_c1' {
	local rowname: word `i' of `outcomes_c1'
	matrix panel_a[`i', 1] = e(rw_`rowname')
}

**FOR COLUMN 1: NEED TO FILL IN P-VALUES FOR VARIABLES WITH NO REGRESSIONS
matrix panel_a[3, 1] = 9999 //dummy value to be skipped in table output 


//column 2
//keep only survey_round == 2 for faster runtime 
restore
preserve
keep if survey_round == 2

//Column 2 (6-year, Lottery Only)
rwolf_pkh `outcomes_c2', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(38102031) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panela_count_c2' {
	local rowname: word `i' of `outcomes_c2'
	matrix panel_a[`i', 2] = e(rw_`rowname')
}

restore




*PANEL B
preserve 
keep if survey_round == 1

//local with outcome variables
local outcomes_c1 wageonly_16to17 wageonly_20hrs_16to17 
local outcomes_c2 wageonly_16to17 wageonly_20hrs_16to17 wageonly_18to21 wageonly_20hrs_18to21 
matrix rownames panel_b = `outcomes_c2'

//Column 1 (2-year, Lottery Only)
rwolf_pkh `outcomes_c1', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(84840193) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelb_count_c1' {
	local rowname: word `i' of `outcomes_c1'
	matrix panel_b[`i', 1] = e(rw_`rowname')
}

**FOR COLUMN 1: NEED TO FILL IN P-VALUES FOR VARIABLES WITH NO REGRESSIONS
matrix panel_b[3, 1] = 9999 //dummy value to be skipped in table output 
matrix panel_b[4, 1] = 9999


//Column 2 (6-year, Lottery Only)
//keep only survey_round == 2 for faster runtime 
restore
preserve
keep if survey_round == 2

rwolf_pkh `outcomes_c2', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(12746122) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelb_count_c2' {
	local rowname: word `i' of `outcomes_c2'
	matrix panel_b[`i', 2] = e(rw_`rowname')
}

restore






****
*PANEL C
cd "`PKH'data/coded"
use "child6to15_fertility_marriage_outcomes.dta", clear

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


//specify outcomes for column 2
local outcomes_c2 married_16to17 married_18to21
matrix rownames panel_c = `outcomes_c2'


//NOTE: NO R-W ADJUSTMENT NEEDED BECAUSE ONLY ONE OUTCOME
//Column 1
ivregress 2sls married_16to17 `baseline_controls' kabu_* (pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)
matrix panel_c[1, 1] = (2 * ttail(e(df_m), abs(_b[pkh_by_this_wave]/_se[pkh_by_this_wave])))


**FOR COLUMN 1: NEED TO FILL IN P-VALUES FOR VARIABLES WITH NO REGRESSIONS
matrix panel_c[2, 1] = 9999 //dummy value to be skipped in table output 


//Column 2 (6-year, Lottery Only)
preserve
keep if survey_round == 2
rwolf_pkh `outcomes_c2', indepvar(pkh_by_this_wave) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(90029384) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelc_count_c2' {
	local rowname: word `i' of `outcomes_c2'
	matrix panel_c[`i', 2] = e(rw_`rowname')
}


restore



***Export appended matrix***
//change to output directory 
cd "`output'"

matrix pvals = panel_a \ panel_b \ panel_c
putexcel A1 = matrix(pvals), names nformat(number_d3)





