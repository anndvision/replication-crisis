*** 1. INTRODUCTION ***
/*
Description: Calculates Romano-Wolf p-values for Table 7
Uses: 	household_allwaves_master.dta
Creates: rwolf_pvalues_table7.xls
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


*** 2. TABLE 7 (HOUSEHOLD ECONOMIC OUTCOMES) ***
//specify file for putexcel
putexcel set rwolf_pvalues_table7.xls, sheet(table7) replace

//set local with number of outcomes for each panel 
local panela_count = 1
local panelb_count = 3
//declare matrices for storing p-values
matrix panel_a = J(`panela_count', 2, .)
matrix panel_b = J(`panelb_count', 2, .)

**Data for all panels**
cd "`PKH'data/coded"
use "household_allwaves_master.dta", clear

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
keep  logpcexp logpcexp_food logpcexp_alctobacco logpcexp_educ_health logpcexp_milk_egg ///
	owns_any_land head_employed total_livestock_owned ///
	`baseline_controls' `interactions' kabu* kecamatan survey_round pkh_by_this_wave L07


//change to output directory and run RW 
cd "`output'"

***PANEL A: OVERALL EXPENDITURE***
matrix rownames panel_a = logpcexp

//NOTE: NO R-W ADJUSTMENT NEEDED BECAUSE ONLY ONE OUTCOME
//Column 1
ivregress 2sls logpcexp `baseline_controls' kabu_* (pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)
matrix panel_a[1, 1] = (2 * ttail(e(df_m), abs(_b[pkh_by_this_wave]/_se[pkh_by_this_wave])))

//Column 2
ivregress 2sls logpcexp `baseline_controls' kabu_* (pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)
matrix panel_a[1, 2] = (2 * ttail(e(df_m), abs(_b[pkh_by_this_wave]/_se[pkh_by_this_wave])))





***PANEL B: Household Land + Livestock investment
local outcomes owns_any_land head_employed total_livestock_owned  
matrix rownames panel_b = `outcomes'

//keep only survey_round == 1 for faster runtime 
preserve
keep if survey_round == 1

//Column 1 (2-year, Lottery Only)
rwolf_pkh_hh `outcomes', indepvar(pkh_by_this_wave) expvars(logpcexp logpcexp_food logpcexp_alctobacco logpcexp_educ_health logpcexp_milk_egg) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(93138212) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelb_count' {
	local rowname: word `i' of `outcomes'
	matrix panel_b[`i', 1] = e(rw_`rowname')
}

//Column 3 (6-year, Lottery Only)
restore
preserve
keep if survey_round == 2
rwolf_pkh_hh `outcomes', indepvar(pkh_by_this_wave) expvars(logpcexp logpcexp_food logpcexp_alctobacco logpcexp_educ_health logpcexp_milk_egg) ///
	method(ivregress) iv(L07) strata(kabupaten) cluster(kecamatan) reps(1000) seed(11271283) controls(`baseline_controls' kabu_*) vce(cluster kecamatan) verbose

forvalues i = 1/`panelb_count' {
	local rowname: word `i' of `outcomes'
	matrix panel_b[`i', 2] = e(rw_`rowname')
}

restore



***Export appended matrix***
matrix pvals = panel_a \ panel_b
putexcel A1 = matrix(pvals), names nformat(number_d3)







