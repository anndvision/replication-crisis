
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Appendix Table 2
Uses: 			marriedwoman16to49_allwaves_master.dta, infantmortality_allwaves_master
Creates: 		appendix_table2.tex
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
local latex = "`output'/latex"
local TARGET2 = "$TARGET2"

*** 5. PANEL A REGRESSIONS (MISCARRIAGES/STILLBIRTH) ***
cd "`PKH'data/coded"
use "marriedwoman16to49_allwaves_master.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics (INCLUDES LOG PCE)
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm hhsize_ln_baseline_nm logpcexp_baseline_nm *miss


//generate interaction variables
drop *_i //drop 2 existing panel variables ending in *_i
foreach x of varlist `baseline_controls' {
	generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)


*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Number of pre-natal visits 
//Midline
eststo model_a: ivregress 2sls miscarriage_or_stillborn `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ miscarriage_or_stillborn if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline
eststo model_b: ivregress 2sls miscarriage_or_stillborn `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ miscarriage_or_stillborn if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table2.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome:"))
	varlab(pkh_by_this_wave "Miscarriage or stillbirth in last 24 months")
	refcat(pkh_by_this_wave "\midrule \emph{Panel A: Pregnancy Outcomes}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 



*** 3. PANEL B REGRESSIONS (INFANT MORTALITY) ***
cd "`PKH'data/coded"
use "infantmortality_allwaves_master.dta", clear

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

//and dummies for month-of-birth controls 
tab birth_mon if survey_round == 1, gen(month_dummy_w3_)
tab birth_mon if survey_round == 2, gen(month_dummy_w4_)

//Run regressions
cd "`output'"


**Outcome: Child 0-28 days died in last 24 months
//Midline 
eststo panelb_model_a: ivregress 2sls dead_0to28_days_last24 `baseline_controls' month_dummy_w3_* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ dead_0to28_days_last24 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline 
eststo panelb_model_b: ivregress 2sls dead_0to28_days_last24 `baseline_controls' month_dummy_w4_* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ dead_0to28_days_last24 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


**Outcome: child 1-12 months died in last 24 months
local depvar = "dead_1to12_mons_last24"

//Midline
eststo model11: ivregress 2sls `depvar' `baseline_controls' month_dummy_w3_* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ `depvar' if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'		

//Endline
eststo model12: ivregress 2sls `depvar' `baseline_controls' month_dummy_w4_* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)	

	//add control mean 
	summ `depvar' if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'	



*** 4. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b
	using "appendix_table2.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Child 0-28 days died in last 24 months")
	refcat(pkh_by_this_wave "\\ \emph{Panel B: Infant Mortality}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 



local depvar = "Child 1-12 months died in last 24 months" 
#delimit ;

esttab model11 model12
	using "appendix_table2.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	varlab(pkh_by_this_wave "\\ `depvar'")
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nomtitles
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr




*** 3. PANEL C REGRESSIONS: Fertility Timing ***
cd "`PKH'data/coded"
use "mothers_fertility_timing.dta", clear

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

//get rid of negative values of number of births since baseline
replace births_since_baseline_all = 0 if births_since_baseline_all < 0
replace births_since_baseline_nosbmc = 0 if births_since_baseline_nosbmc < 0

*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Number of non-immediate children in household
//Midline
eststo model_a: ivregress 2sls births_since_baseline_all_v2 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)


	//add control mean 
	summ births_since_baseline_all_v2 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline
eststo model_b: ivregress 2sls births_since_baseline_all_v2 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ births_since_baseline_all_v2 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'



**Outcome: Births since baseline (no stillbirth/miscarriage)
//Midline
eststo model11: ivregress 2sls births_since_baseline_nosbmc_v2 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)


	//add control mean 
	summ births_since_baseline_nosbmc_v2 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Endline
eststo model12: ivregress 2sls births_since_baseline_nosbmc_v2 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)


	//add control mean 
	summ births_since_baseline_nosbmc_v2 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'



*** 4. PANEL C OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table2.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Pregnancies since baseline (all pregnancies)")
	refcat(pkh_by_this_wave "\\ \emph{Panel C: Fertility Timing}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

local depvar = "Pregnancies since baseline (no stillbirths/miscarriages)"

#delimit ;

esttab model11 model12
	using "appendix_table2.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	varlab(pkh_by_this_wave "\\ `depvar'")
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nomtitles
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr






