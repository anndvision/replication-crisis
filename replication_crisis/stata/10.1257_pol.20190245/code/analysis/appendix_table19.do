
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


//make necessary combinations of food consumption outcomes 
gen ate_beef_or_pork = (ate_beef == 1 | ate_pork == 1)
replace ate_beef_or_pork = . if missing(ate_beef) & missing(ate_pork)

gen ate_chicken_duck_fish = (ate_chicken_duck == 1 | ate_fish == 1)
replace ate_chicken_duck_fish = . if missing(ate_chicken_duck) & missing(ate_fish)

gen ate_other_grain_or_noodle = (ate_other_grain == 1 | ate_inst_noodle == 1)
replace ate_other_grain_or_noodle = . if missing(ate_other_grain) & missing(ate_inst_noodle)

gen ate_fruit_veg_tuber = (ate_veg == 1 | ate_fruit == 1 | ate_tubers == 1)
replace ate_fruit_veg_tuber = . if missing(ate_veg) & missing(ate_fruit) & missing(ate_tubers)

gen ate_snack_sweets = (ate_snack == 1 | ate_sweets == 1)
replace ate_snack_sweets = . if missing(ate_snack) & missing(ate_sweets)


***RUN REGRESSIONS RESTRICTING TO <5 YEARS ONLY***
*** 4. PANEL A REGRESSIONS ***
cd "`output'"

**Outcome: Stunting
//Midline, Lottery Only 
eststo model_a: ivregress 2sls ate_milk `baseline_controls' agebin* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 1 & age_months <= 60, vce(cluster kecamatan)

	//add control mean 
	summ ate_milk if L07 == 0 & survey_round == 1 & e(sample) == 1 & age_months <= 60
	estadd scalar control_mean `r(mean)'

//Endline, Lottery Only 
eststo model_b: ivregress 2sls ate_milk `baseline_controls' agebin* kabu_* /// 
	(pkh_by_this_wave = L07) if survey_round == 2 & age_months <= 60, vce(cluster kecamatan)

	//add control mean 
	summ ate_milk if L07 == 0 & survey_round == 2 & e(sample) == 1 & age_months <= 60
	estadd scalar control_mean `r(mean)'


**Outcomes: food consumption categories
forvalues i = 1/7 {
	local depvar: word `i' of "ate_egg" "ate_beef_or_pork" "ate_chicken_duck_fish" "ate_rice" "ate_other_grain_or_noodle" ///
					"ate_fruit_veg_tuber" "ate_snack_sweets"
	
	forvalues j = 1/2 {
		eststo model`i'`j': ivregress 2sls `depvar' `baseline_controls' agebin* kabu_* /// 
		(pkh_by_this_wave = L07) if survey_round == `j' & age_months <= 60, vce(cluster kecamatan)

		//add control mean 
		summ `depvar' if L07 == 0 & survey_round == `j' & e(sample) == 1 & age_months <= 60
		estadd scalar control_mean `r(mean)'
	}

}


*** 5. PANEL A OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model_a model_b
	using "appendix_table19.tex", booktabs label se
	keep(pkh_by_this_wave)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year", lhs("Outcome:"))
	varlab(pkh_by_this_wave "\midrule Drank milk last week")
	stats(control_mean, labels(" ") fmt("%12.3f"))
	nogaps
	noobs
	nonotes
	nolines
	fragment
	replace;

#delimit cr 

forvalues i = 1/7 {
	local depvar: word `i' of "Ate egg last week" "Ate beef or pork last week" "Ate chicken, duck, or fish last week" ///
					"Ate rice last week" "Ate other grain or noodles last week" "Ate fruit, vegetables, or tubers last week" ///
					"Ate snacks or sweets last week"
	
	#delimit ;

	esttab model`i'1 model`i'2
		using "appendix_table19.tex", booktabs label se
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





