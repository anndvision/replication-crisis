
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Table 4 (Health Outcomes)
Uses: 			child0to36months_allwaves_master.dta
Creates: 		table4.tex
*/

*** 2. SET ENVIRONMENT ***
version 14.1 		
set more off		
set matsize 10000
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


*** 3. IMPORT ADJUSTED P-VALUES FROM R-W OUTPUT ***
cd "`PKH'output"
//import excel document 
import excel using "rwolf_pvalues_table4.xls", firstrow clear
//loop over number of variables 
qui count 
local num_vars = `r(N)'
//loop for each variable
forv i = 1/`num_vars' {
	local var_name = A[`i']
	//loop for each column
	forv j = 1/2 {
		local `var_name'_p_`j': di %05.3f c`j'[`i'] // round to 3 decimal places
		//change 0 to <0.001
		if ``var_name'_p_`j'' == 0 | "``var_name'_p_`j''" == "0.000" {
			local `var_name'_p_`j' = "[" + "<0.001" + "]"
		}
		else {
			local `var_name'_p_`j' = "[" + "``var_name'_p_`j''" + "]"
		}
	}
}


*** 4. PREPARE FOR REGRESSION ***
cd "`PKH'data/coded"
use "child0to36months_allwaves_master.dta", clear

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


//now also generate interactions with survey round variable for "stack and cluster" approach 
//to calculating 2-vs-6-year p-values
//generate dummy for survey round 
gen survey_round_1 = (survey_round == 1) if survey_round != 0
gen survey_round_2 = (survey_round == 2) if survey_round != 0

//baseline controls, endogenous regressor, and instrument 
foreach var of varlist `baseline_controls' {
	gen `var'1 = `var'*survey_round_1
	gen `var'2 = `var'*survey_round_2

	local baseline_controls_1 `baseline_controls_1' `var'1
	local baseline_controls_2 `baseline_controls_2' `var'2
}

//fixed effects / age bins 
foreach var of varlist kabu_* agebin* {
	gen d`var'_1 = `var'*survey_round_1
	gen d`var'_2 = `var'*survey_round_2
}

//endogenous regressors (don't include in loop above so that they don't get added to baseline controls local)
foreach var of varlist pkh_by_this_wave L07 {
	gen `var'1 = `var'*survey_round_1
	gen `var'2 = `var'*survey_round_2
}


*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Stunting
//Midline, Lottery Only 
eststo model_a: ivregress 2sls mal_heightforage `baseline_controls' agebin* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`mal_heightforage_p_1'"

	//add control mean 
	summ mal_heightforage if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo model_b: ivregress 2sls mal_heightforage `baseline_controls' agebin* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add RW adjusted p-value
	estadd local rwpval "`mal_heightforage_p_2'"

	//add control mean 
	summ mal_heightforage if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Calculate 2-vs-6-year p-value! 
eststo pval_26: ivregress 2sls mal_heightforage `baseline_controls_1' `baseline_controls_2' /// 
	dagebin* dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan) 

	//test 2 vs. 6 year 
	test pkh_by_this_wave1=pkh_by_this_wave2
	estadd scalar pval26 `r(p)'


**Outcomes: severe_heightforage, mal_weightforage, severe_weightforage, 
forvalues i = 1/3 {
	local depvar: word `i' of "severe_heightforage" "mal_weightforage" "severe_weightforage" 
	
	forvalues j = 1/2 {

		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' agebin* kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j', vce(cluster kecamatan)

		//add RW adjusted p-value
		estadd local rwpval "``depvar'_p_`j''"

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
		estadd scalar control_mean `r(mean)'

		//Calculate 2-vs-6-year p-value! 
		eststo pval_26_`i': ivregress 2sls `depvar' `baseline_controls_1' `baseline_controls_2' /// 
		dagebin* dkabu_* (pkh_by_this_wave? = L07?), vce(cluster kecamatan)

		//test 2 vs 6 year 
		test pkh_by_this_wave1 = pkh_by_this_wave2
		estadd scalar pval26 `r(p)'	
	}

}


*** 5. OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b pval_26
	using "table4.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year" "p-value (2-Yr. = 6-Yr.)", lhs("Outcome:"))
	varlab(pkh_by_this_wave "\midrule Stunted")
	scalars("rwpval \ " "control_mean \ " "pval26 \ ")
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/3 {
	local depvar: word `i' of "Severely stunted" "Malnourished" ///
								"Severely malnourished"
	
	#delimit ;

	esttab model`i'1 model`i'2 pval_26_`i'
		using "table4.tex", booktabs label se
		keep(pkh_by_this_wave)
		b(%12.3f)
		se(%12.3f)
		nostar
		varlab(pkh_by_this_wave "\\ `depvar'")
		scalars("rwpval \ " "control_mean \ " "pval26 \ ")
		nomtitles
		nogaps
		nonumbers
		noobs
		nonotes
		nolines
		fragment
		append;

		#delimit cr
}

#delimit cr


