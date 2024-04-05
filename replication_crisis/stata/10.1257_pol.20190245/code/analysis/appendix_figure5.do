*** 1. INTRODUCTION ***
/*
Description: Creates graph of non-parametric estimation of stunting effects
Uses: 		
Creates: appendix_figure_stunting.png, appendix_figure_severe_stunting.png
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
cd "`PKH'"

//Set working directory:
cd "`pkhdata'"

* Declare constants here
local np = 								50 // # of points at which to calculate
local bw = 								4

* Declare output file paths
local output_dir = 						"`PKH'output/graphs/"


******************************************************************************
* 1. Dataset prep
******************************************************************************
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

//rename age variable appropriately
rename age age_days

******************************************************************************
* 2. Create graphs
******************************************************************************
//change to graphs output directory 
cd "`output_dir'"

//define our x-variable as age in days 
local agedef = "days" 
local agevar = "age_`agedef'"

//one loop for each anthropometric outcome 
forvalues j = 1/2 {

	//set dependent variable 
	local depvar: word `j' of "mal_heightforage" "severe_heightforage" 
	local outcome_label: word `j' of "stunting" "severe stunting" 
	local outcome_file_label: word `j' of "stunting" "severe_stunting" 

	//set survey round 
	local period = "6-year follow-up"

	* Range of independent variable for particular survey round if not missing anthropometric outcome
	summ `agevar' if survey_round == 2 & !missing(`depvar')
	local xmin = r(min)
	local xmax = r(max)

	* Calculate parameters for fan reg
	local gsize = `np' + 1 // Number of points at which to calculate, 50 to 100 are typically fine 
	local st = (`xmax' - `xmin')/(`gsize'-1) // Size of each step
	local h = (`xmax' - `xmin') / `bw' // Bandwidth - equal to 1-Nth of total distance 

	* Generate the estimated function (3) and its derivative (4)
	gen predicted_beta = .
	gen predicted_se = .
	gen predicted_xvals = .
		
	* Loop until reaching the last cell of the grid
	forval ic = 1/`gsize' {	
		
		* Display the counter 
		dis `ic'	
		
		quietly {
			
			* Get the ic entry in the grid *
			local xx = `xmin'+`st'*(`ic'-1)	
			
			* Triangle Kernel with bandwidth given by h
			gen kz = max(1 - abs(`agevar'-`xx')/`h',0)
		
			* Perform the regression weighted by the kernel (analogous to GLS) 
			capture ivregress 2sls `depvar' `baseline_controls' agebin* kabu_* /// 
			(pkh_by_this_wave = L07) [aw=kz] if survey_round == 2 & kz != ., vce(cluster kecamatan)
			
			* The estimated regression is the value at x
			capture replace predicted_beta = _b[pkh_by_this_wave] in `ic'
			capture replace predicted_se = _se[pkh_by_this_wave] in `ic'
			capture replace predicted_xvals = `xx' in `ic'
			drop kz	
		}	
	}
	
	gen y_1 = predicted_beta + 1.96*predicted_se
	gen y_2 = predicted_beta - 1.96*predicted_se

	//horizontal line for graphing
	gen zero_line = 0

	//output graphs
	twoway (rarea y_1 y_2 predicted_xvals, legend(off) lcolor(gs13) fcolor(gs13) xtitle("Child's age (recorded in `agedef')") ytitle("Estimated regression coefficient")) ///
 	(line predicted_beta predicted_xvals, legend(off)) (line zero_line predicted_xvals, legend(off) lcolor(black)), ylabel(#8)  ///
 	yscale(range(-0.3 0.15)) xlabel(365.25 "1 year" 730.5 "2 years" 1095.75 "3 years" 1461 "4 years" 1826.25 "5 years") ///
 	title("Treatment effect on `outcome_label', `period'", size(medsmall))

	graph export "appendix_figure5_`outcome_file_label'.png", replace
	graph export "appendix_figure5_`outcome_file_label'.eps", replace
	
	drop predicted_beta predicted_se predicted_xvals y_1 y_2 zero_line
}






