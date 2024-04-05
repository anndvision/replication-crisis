

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
use "tracking_child6to15_attrition.dta", clear

//drop *_miss variables related to food/non-food expenditure
drop logpcexp_food_baseline_miss logpcexp_nonfood_baseline_miss

//create local with all baseline characteristics
local baseline_controls /// 
	  hh_head_agr_baseline_nm hh_head_serv_baseline_nm hh_educ*baseline_nm /// 
	  roof_type*baseline_nm wall_type*baseline_nm floor_type*baseline_nm clean_water_baseline_nm ///
	  own_latrine_baseline_nm square_latrine_baseline_nm own_septic_tank_baseline_nm /// 
	  electricity_PLN_baseline_nm logpcexp_baseline_nm hhsize_ln_baseline_nm *miss

//generate interaction variables
foreach x of varlist `baseline_controls' {
	generate `x'_i = `x'*L07
}

//create local with all interactions 
local interactions *_i

//generate indicator variables for kabupaten fixed effects
tab kabupaten, gen(kabu_)

*** 4. ATTRITION REGRESSIONS ***
cd "`output'"

//Wave III / midline 
eststo model1: reg not_surveyed L07 `baseline_controls' kabu_* if survey_round == 1, vce(cluster kecamatan)

//add control mean
summ not_surveyed if L07 == 0 & survey_round == 1 & e(sample) == 1
estadd scalar control_mean `r(mean)'

//Wave IV / endline
eststo model2: reg not_surveyed L07 `baseline_controls' kabu_* if survey_round == 2, vce(cluster kecamatan)

//add control mean
summ not_surveyed if L07 == 0 & survey_round == 2 & e(sample) == 1
estadd scalar control_mean `r(mean)'

//Wave IV / BOYS ONLY 
eststo model3: reg not_surveyed L07 `baseline_controls' kabu_* if survey_round == 2 & gender_baseline == 1, vce(cluster kecamatan)

//add control mean
summ not_surveyed if L07 == 0 & survey_round == 2 & e(sample) == 1 & gender_baseline == 1
estadd scalar control_mean `r(mean)'

//Wave IV / GIRLS ONLY 
eststo model4: reg not_surveyed L07 `baseline_controls' kabu_* if survey_round == 2 & gender_baseline == 0, vce(cluster kecamatan)

//add control mean
summ not_surveyed if L07 == 0 & survey_round == 2 & e(sample) == 1 & gender_baseline == 0
estadd scalar control_mean `r(mean)'



*** 5. OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;
esttab model1 model2 model3 model4
	using "appendix_table3.tex", booktabs label se
	keep(L07)
	b(%12.3f)
	se(%12.3f)
	nostar
	mlabels("2-Year" "6-Year" "6-Year" "6-Year", lhs("Outcome:"))
	mgroups("Full Sample" "Boys Only" "Girls Only", pattern(1 0 1 1) span prefix(\multicolumn{@span}{c}{) suffix(}) erepeat(\cmidrule(lr){@span}))
	varlab(L07 "Lost to Follow-Up")
	nogaps
	scalars("control_mean Control Mean")
	nonotes
	fragment
	replace;

#delimit cr


