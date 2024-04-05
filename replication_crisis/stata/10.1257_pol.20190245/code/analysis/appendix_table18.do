
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


*** 3. PREPARE FOR REGRESSION ***
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

drop *_i 

//generate interaction variables
foreach x of varlist `baseline_controls' {
	generate `x'_i = `x'*L07
}
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)

*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**restrict "knows birthweight" variable to <35 as well
gen knows_birthweight_35 = knows_birthweight if iir01 < 35

**Outcome: Mother's composite knowledge
//Midline
eststo model_a: ivregress 2sls mother_knowledge_35 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1 & ich02_line != 2 & ich02_line != 3, vce(cluster kecamatan) //don't double-count mothers with multiple pregnancies


	//add control mean 
	summ mother_knowledge_35 if L07 == 0 & survey_round == 1 & e(sample) == 1 & ich02_line != 2 & ich02_line != 3
	estadd scalar control_mean `r(mean)'

//Endline
eststo model_b: ivregress 2sls mother_knowledge_35 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & ich02_line != 2 & ich02_line != 3, vce(cluster kecamatan)


	//add control mean 
	summ mother_knowledge_35 if L07 == 0 & survey_round == 2 & e(sample) == 1 & ich02_line != 2 & ich02_line != 3
	estadd scalar control_mean `r(mean)'


**Outcomes: breastfeed_correct_pct, diarrhea_correct_pct, pre_natal_question_correct
forvalues i = 1/4 {
	local depvar: word `i' of "breastfeed_correct_pct_35" "diarrhea_correct_pct_35" "pre_natal_question_correct_35" "knows_birthweight_35"
	
	forvalues j = 1/2 {
		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j' & ich02_line != 2 & ich02_line != 3, vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1 & ich02_line != 2 & ich02_line != 3
		estadd scalar control_mean `r(mean)'
	}

}


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table18.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome:"))
	varlab(pkh_by_this_wave "Mother's composite knowledge \% (of 5)")
	refcat(pkh_by_this_wave "\midrule \emph{Panel A: Maternal Knowledge of Proper Health Practices (Ages <35)}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/4 {
	local depvar: word `i' of "\% of breastfeeding questions correct (of 2)" "\% of diarrhea questions correct (of 2)" ///
								"Pre-natal visits question correct" "Knows child's birth weight"
	
	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table18.tex", booktabs label se
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
}

#delimit cr



*** 6. PANEL B REGRESSIONS ***
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
cd "`output'"


**Outcome: Ever been breastfed
//Midline 
eststo panelb_model_a: ivregress 2sls ever_breastfed `baseline_controls' agebin* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1 & age_months <= 60, vce(cluster kecamatan)

	//add control mean 
	summ ever_breastfed if L07 == 0 & survey_round == 1 & e(sample) == 1 & age_months <= 60
	estadd scalar control_mean `r(mean)'


//Endline
eststo panelb_model_b: ivregress 2sls ever_breastfed `baseline_controls' agebin* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & age_months <= 60, vce(cluster kecamatan)

	//add control mean 
	summ ever_breastfed if L07 == 0 & survey_round == 2 & e(sample) == 1 & age_months <= 60
	estadd scalar control_mean `r(mean)'


**Outcomes: Exclusively breastfed for 3 months
forvalues i = 1/1 {
	local depvar: word `i' of "excl_breastfed_3mon"
	
		forvalues j = 1/2 {
		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' agebin* kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j' & age_months <= 60, vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1 & age_months <= 60
		estadd scalar control_mean `r(mean)'
	}

}


*** 7. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b
	using "appendix_table18.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Child ever been breastfed")
	refcat(pkh_by_this_wave "\emph{Panel B: Children's Breastfeeding Outcomes (0-60 Months)}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

forvalues i = 1/1 {
	local depvar: word `i' of "Exclusively breastfed for 3 months after birth"
	
	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table18.tex", booktabs label se
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
}

#delimit cr



*** 8. PANEL C REGRESSIONS ***
cd "`PKH'data/coded"
use "household_allwaves_master.dta", clear

//drop above 99th percentile values for pcexp by survey_round
drop if survey_round == 1 & pcexp >= 748321.4
drop if survey_round == 2 & pcexp >= 1323839

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
cd "`output'"


**Outcome: Household has piped water
//Midline, Lottery Only 
eststo panelc_model_a: ivregress 2sls clean_water `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ clean_water if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo panelc_model_b: ivregress 2sls clean_water `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ clean_water if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


**Outcomes: Latrine, square latrine, own septic tank, electricity from PLN 
forvalues i = 1/4 {
	local depvar: word `i' of "own_latrine" "square_latrine" "own_septic_tank" "electricity_PLN"
	
	forvalues j = 1/2 {
		eststo panelc_model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}

}


*** 9. PANEL C OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelc_model_a panelc_model_b
	using "appendix_table18.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Household has piped water")
	refcat(pkh_by_this_wave "\emph{Panel C: Household Investment in Sanitation}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

forvalues i = 1/4 {
	local depvar: word `i' of "Household has own latrine" "Household has square latrine" "Household has own septic tank" "Household has PLN electricity"
	
	#delimit ;

	esttab panelc_model`i'1 panelc_model`i'2
		using "appendix_table18.tex", booktabs label se
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
}

#delimit cr

