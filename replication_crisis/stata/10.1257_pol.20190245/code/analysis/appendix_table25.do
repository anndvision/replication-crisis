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

**Outcome: Enrollment in any school (age 15)
//Midline, Lottery Only 
eststo model_a: ivregress 2sls enroll_any_age15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ enroll_any_age15 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


//Endline, Lottery Only 
eststo model_b: ivregress 2sls enroll_any_age15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ enroll_any_age15 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


**Outcomes: ages 16, 17, 18
forvalues i = 1/3 {
	local depvar: word `i' of "enroll_any_age16" "enroll_any_age17" "enroll_any_age18"
	
	forvalues j = 1/2 {

	if "`depvar'" == "enroll_any_age18" & `j' == 1 {
		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave_new = L07) if survey_round == `j', vce(cluster kecamatan)
	}
	else {
		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j', vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}
	
	}
}


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table25.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome:"))
	varlab(pkh_by_this_wave "Enrolled in school (Age 15)")
	refcat(pkh_by_this_wave "\midrule \emph{Panel A: School Enrollment (Any Level) by Age}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 


forvalues i = 1/3{
	local depvar: word `i' of "Enrolled in school (Age 16)" "Enrolled in school (Age 17)" "Enrolled in school (Age 18)"
		
	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table25.tex", booktabs label se
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

*** 4. PANEL B REGRESSIONS ***
cd "`output'"

**Outcome: Enrollment in high school (age 15)
//Midline, Lottery Only 
eststo panelb_model_a: ivregress 2sls enroll_SMA_15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1, vce(cluster kecamatan)

	//add control mean 
	summ enroll_SMA_15 if L07 == 0 & survey_round == 1 & e(sample) == 1
	estadd scalar control_mean `r(mean)'

//Endline, Lottery Only 
eststo panelb_model_b: ivregress 2sls enroll_SMA_15 `baseline_controls' kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2, vce(cluster kecamatan)

	//add control mean 
	summ enroll_SMA_15 if L07 == 0 & survey_round == 2 & e(sample) == 1
	estadd scalar control_mean `r(mean)'


**Outcomes: ages 16, 17, 18
forvalues i = 1/3 {
	local depvar: word `i' of "enroll_SMA_16" "enroll_SMA_17" "enroll_SMA_18"
	
	forvalues j = 1/2 {

	if "`depvar'" == "enroll_SMA_18" & `j' == 1 {
		eststo panelb_model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave_new = L07) if survey_round == `j', vce(cluster kecamatan)
	}
	else {
		eststo panelb_model`i'`j': ivregress 2sls `depvar' `baseline_controls' kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j', vce(cluster kecamatan)

			//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1
		estadd scalar control_mean `r(mean)'
	}
	

	}
}


*** 5. PANEL B OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab panelb_model_a panelb_model_b
	using "appendix_table25.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	nomtitles
	varlab(pkh_by_this_wave "Enrolled in high school (Age 15)")
	refcat(pkh_by_this_wave "\\ \emph{Panel B: High School Enrollment by Age}", nolabel)
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	nonumbers
	noobs
	nonotes
	nolines
	fragment
	append;

#delimit cr 


forvalues i = 1/3{
	local depvar: word `i' of "Enrolled in high school (Age 16)" "Enrolled in high school (Age 17)" "Enrolled in high school (Age 18)"
		
	#delimit ;

	esttab panelb_model`i'1 panelb_model`i'2
		using "appendix_table25.tex", booktabs label se
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

