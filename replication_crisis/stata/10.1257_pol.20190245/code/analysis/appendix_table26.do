
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

//generate dummy version of PKH variable to be able to omit certain output 
gen pkh_by_this_wave_new = pkh_by_this_wave

*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Enrollment in any school
//Endline, boys
eststo model_a: ivregress 2sls enroll_any_15to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 1, vce(cluster kecamatan)

	//add control mean 
	summ enroll_any_15to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 1
	estadd scalar control_mean `r(mean)'

//Endline, girls
eststo model_b: ivregress 2sls enroll_any_15to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 3, vce(cluster kecamatan)

	//add control mean 
	summ enroll_any_15to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 3
	estadd scalar control_mean `r(mean)'


**Outcome: completed high school 18-21,....
forvalues i = 1/2 {	
	local depvar: word `i' of "enroll_SMA_15to17" "completed_SMA_18to21" ///

	forvalues j = 1/2 {
		if `j' == 1 {
			local gender = "1"
		}
		else {
			local gender = "3"
		}

	eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == `gender', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == `gender'
		estadd scalar control_mean `r(mean)'
	
	}

}


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table26.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("Boys" "Girls", lhs("Outcome:"))
	varlab(pkh_by_this_wave "Enrolled in school (Ages 15-17)")
	refcat(pkh_by_this_wave "\midrule \emph{Panel A: School Enrollment/Completion Outcomes}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/2 {
	local depvar: word `i' of "Enrolled in high school (Ages 15-17)" "Completed high school (Ages 18-21)" ///

	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table26.tex", booktabs label se
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
cd "`PKH'output"

**Outcome: wageonly_16to17
//Endline, boys
eststo panelb_model_a: ivregress 2sls wageonly_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 1, vce(cluster kecamatan)

	//add control mean 
	summ wageonly_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 1
	estadd scalar control_mean `r(mean)'


//Endline, girls
eststo panelb_model_b: ivregress 2sls wageonly_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == 3, vce(cluster kecamatan)

	//add control mean 
	summ wageonly_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & air02 == 3
	estadd scalar control_mean `r(mean)'


**Outcome: 20+ hours (16-17), any wage work 18-21, wage work 20+ hours 18-21
forvalues i = 1/3 {
	local depvar: word `i' of "wageonly_20hrs_16to17" "wageonly_18to21" "wageonly_20hrs_18to21"
	
	forvalues j = 1/2 {
		if `j' == 1 {
			local gender = "1"
		}
		else {
			local gender = "3"
		}


	eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & air02 == `gender', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == 2 & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}
}


*** 7. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b
	using "appendix_table26.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Worked for wage last month (Ages 16-17)")
	refcat(pkh_by_this_wave "\\ \emph{Panel B: Labor Outcomes (Ages 16-21)}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

forvalues i = 1/3 {
	local depvar: word `i' of "Worked 20+ hours for wage last month (Ages 16-17)" "Worked for wage last month (Ages 18-21)" ///
								"Worked 20+ hours for wage last month (Ages 18-21)"
	
	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table26.tex", booktabs label se
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

//generate dummy version of PKH variable to be able to omit certain output 
gen pkh_by_this_wave_new = pkh_by_this_wave

cd "`output'"

**Outcome: Married (Age 16-17)
//Endline, boys
eststo panelc_model_a: ivregress 2sls married_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & gender_current_wave == 1, vce(cluster kecamatan)

	
	//add control mean 
	summ married_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & gender_current_wave == 1
	estadd scalar control_mean `r(mean)'

//Endline, girls
eststo panelc_model_b: ivregress 2sls married_16to17 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & gender_current_wave == 0, vce(cluster kecamatan)


	//add control mean 
	summ married_16to17 if L07 == 0 & survey_round == 2 & e(sample) == 1 & gender_current_wave == 0
	estadd scalar control_mean `r(mean)'



**Outcome: married_18to21
	local depvar = "married_18to21"
	
	forvalues j = 1/2 {
		if `j' == 1 {
			local gender = "1"
		}
		else {
			local gender = "0"
		}

	eststo modelc1`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & gender_current_wave == `gender', vce(cluster kecamatan)


		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == 2 & e(sample) == 1 & gender_current_wave == `gender'
		estadd scalar control_mean `r(mean)'
	
	}


*** 9. PANEL C OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelc_model_a panelc_model_b
	using "appendix_table26.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Married (Ages 16-17)")
	refcat(pkh_by_this_wave "\\ \emph{Panel C: Marriage Outcomes (Ages 16-21)}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 

local depvar = "Married (Ages 18-21)"
	
#delimit ;

esttab modelc11 modelc12
	using "appendix_table26.tex", booktabs label se
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

