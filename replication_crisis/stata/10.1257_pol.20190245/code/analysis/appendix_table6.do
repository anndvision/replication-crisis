
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Appendix Table 6 (Balance check)
Uses: 			marriedwoman16to49_allwaves_master.dta, child0to36months_allwaves_master.dta, 
				child6to15_allwaves_master.dta
Creates: 		appendix_table6.tex
Notes: 			Requires the command "fmttable"
*/



***UNCOMMENT TO INSTALL FMTTABLE COMMAND
//ssc install fmttable

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


*** 3. PREPARE DATA & STORE MATRIX ***
cd "`PKH'data/coded"
//start with first dataset (Mother 16-49)
use "marriedwoman16to49_allwaves_master.dta", clear 

//for kabupaten FE
tab kabupaten, gen(kabu_)

//we only want baseline observations! 
keep if survey_round == 0
drop if missing(L07)

local numvars = 14 //Input number of variables for balance table 
local numcols = 3
local numrows = 2*`numvars' //to add in SEs

//declare necessary matrices, fill with missing values for now
matrix nmeans = J(`numrows',`numcols',.)
matrix NC_b_se = J(`numvars',2,.) //NC for "no controls"
matrix FE_b_se = J(`numvars',2,.) //FE for "fixed effects"
matrix stars_NC = J(`numvars',2,0)
matrix stars_FE = J(`numvars',2,0)
matrix chisq = J(2,5,.) //for joint test later

*Calculate stats for mother variables
forvalues i = 1/4 {
	//set locals for iterating through main matrix 
	local r = 2*`i' - 1
	*local se_row = 2*`i'
	local depvar: word `i' of "pre_natal_visits" "iron_pills_dummy" "good_assisted_delivery" "post_natal_visits"

	//control mean (3 decimal places, maybe change to 2)
	quietly sum `depvar' if L07 == 0
	matrix nmeans[`r',2] = `r(mean)'

	//treatment mean 
	quietly sum `depvar' if L07 == 1
	matrix nmeans[`r',3] = `r(mean)'

	//treatment effect, no controls (but still cluster SE)
	quietly reg `depvar' L07, vce(cluster kecamatan) //store for joint test
	//number of observations
	matrix nmeans[`r',1] = `e(N)'
	//coefficients and standard errors
	matrix NC_b_se[`i',1] = _b[L07]
	matrix NC_b_se[`i',2] = _se[L07]
	//stars
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	if `p' < 0.10 matrix stars_NC[`i',1] = 1
	if `p' < 0.05 matrix stars_NC[`i',1] = 2 
	if `p' < 0.01 matrix stars_NC[`i',1] = 3
	eststo NC_`depvar': quietly reg `depvar' L07 //store for joint test (no cluster)

	//treatment effect, kabupaten FE 
	quietly reg `depvar' L07 kabu_*, vce(cluster kecamatan)
	//coefficients and standard errors
	matrix FE_b_se[`i',1] = _b[L07]
	matrix FE_b_se[`i',2] = _se[L07]
	//stars
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	if `p' < 0.10 matrix stars_FE[`i',1] = 1
	if `p' < 0.05 matrix stars_FE[`i',1] = 2 
	if `p' < 0.01 matrix stars_FE[`i',1] = 3
	eststo FE_`depvar': quietly reg `depvar' L07 kabu_*, vce(cluster kecamatan) //store for joint test (no cluster)
}


//move onto child (0-36 months) dataset
use "child0to36months_allwaves_master.dta", clear 

//for kabupaten FE
tab kabupaten, gen(kabu_)

//we only want baseline observations! 
keep if survey_round == 0
drop if missing(L07)


****
//rename variables that are too long 
rename imm_age_uptak_percent_only imm_age_percent
rename times_weighed_last3months times_weighed
rename vitA_total_6mons_2years vitamin_a_times

****

*Calculate stats for child variables
forvalues i = 1/4 {
	//set locals for iterating through main matrix 
	local r = (2*`i' - 1) + 8
	local k = `i' + 4
	*local se_row = 2*`i'
	local depvar: word `i' of "imm_age_complete" "imm_age_percent" "times_weighed" "vitamin_a_times"

	//control mean (3 decimal places, maybe change to 2)
	quietly sum `depvar' if L07 == 0
	matrix nmeans[`r',2] = `r(mean)'

	//treatment mean 
	quietly sum `depvar' if L07 == 1
	matrix nmeans[`r',3] = `r(mean)'

	//treatment effect, no controls (but still cluster SE)
	quietly reg `depvar' L07, vce(cluster kecamatan)
	//number of observations
	matrix nmeans[`r',1] = `e(N)'
	//coefficients and standard errors
	matrix NC_b_se[`k',1] = _b[L07]
	matrix NC_b_se[`k',2] = _se[L07]
	//stars
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	if `p' < 0.10 matrix stars_NC[`k',1] = 1
	if `p' < 0.05 matrix stars_NC[`k',1] = 2 
	if `p' < 0.01 matrix stars_NC[`k',1] = 3
	eststo NC_`depvar': quietly reg `depvar' L07 //store for joint test (no cluster)


	//treatment effect, kabupaten FE 
	quietly reg `depvar' L07 kabu_*, vce(cluster kecamatan)
	//coefficients and standard errors
	matrix FE_b_se[`k',1] = _b[L07]
	matrix FE_b_se[`k',2] = _se[L07]
	//stars
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	if `p' < 0.10 matrix stars_FE[`k',1] = 1
	if `p' < 0.05 matrix stars_FE[`k',1] = 2 
	if `p' < 0.01 matrix stars_FE[`k',1] = 3
	eststo FE_`depvar': quietly reg `depvar' L07 kabu_* //store for joint test (no cluster)
}

//move onto child (6-15) dataset
use "child6to15_allwaves_master.dta", clear 

//for kabupaten FE
tab kabupaten, gen(kabu_)

//we only want baseline observations! 
keep if survey_round == 0
drop if missing(L07)



*Calculate stats for child variables
forvalues i = 1/6 {
	//set locals for iterating through main matrix 
	local r = (2*`i' - 1) + 16
	local k = `i' + 8
	*local se_row = 2*`i'
	local depvar: word `i' of "enroll_age7to12" "enroll_age13to15" "age7to12_pct_twoweeks" "age13to15_pct_twoweeks" ///
								"age7to12_85_twoweeks" "age13to15_85_twoweeks"

	//control mean (3 decimal places, maybe change to 2)
	quietly sum `depvar' if L07 == 0
	matrix nmeans[`r',2] = `r(mean)'

	//treatment mean 
	quietly sum `depvar' if L07 == 1
	matrix nmeans[`r',3] = `r(mean)'

	//treatment effect, no controls (but still cluster SE)
	eststo NC_`depvar': quietly reg `depvar' L07, vce(cluster kecamatan) //store for joint test
	//number of observations
	matrix nmeans[`r',1] = `e(N)'
	//coefficients and standard errors
	matrix NC_b_se[`k',1] = _b[L07]
	matrix NC_b_se[`k',2] = _se[L07]
	//stars
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	if `p' < 0.10 matrix stars_NC[`k',1] = 1
	if `p' < 0.05 matrix stars_NC[`k',1] = 2 
	if `p' < 0.01 matrix stars_NC[`k',1] = 3

	//treatment effect, kabupaten FE 
	eststo FE_`depvar': quietly reg `depvar' L07 kabu_*, vce(cluster kecamatan) //store for joint test
	//coefficients and standard errors
	matrix FE_b_se[`k',1] = _b[L07]
	matrix FE_b_se[`k',2] = _se[L07]
	//stars
	local p = 2 * ttail(e(df_r),abs(_b[L07] / _se[L07]))
	if `p' < 0.10 matrix stars_FE[`k',1] = 1
	if `p' < 0.05 matrix stars_FE[`k',1] = 2 
	if `p' < 0.01 matrix stars_FE[`k',1] = 3
}



*** 4. OUTPUT TO LATEX ***
matrix rownames nmeans = pre_natal_visits r1 iron_pills_dummy r2 good_assisted_delivery r3 post_natal_visits r4 imm_age_complete r5 imm_age_percent r6 ///
								times_weighed r7 vitamin_a_times r8 enroll_age7to12 r9 enroll_age13to15 r10 age7to12_pct_twoweeks r11 age13to15_pct_twoweeks r12 ///
								age7to12_85_twoweeks r13 age13to15_85_twoweeks r14


//output means matrix first, then differences
cd "`latex'"

#delimit ;

frmttable, statmat(nmeans) 
	sdec(0,3,3)
	rtitles("Pre-natal visits" \ "" \  "90+ iron pills during pregnancy" \ "" \ "Good assisted delivery" \ "" \ "Post-natal visits" \ "" \
			"Immunizations complete for age" \ "" \ "\% of required immunizations completed" \ "" \ "Times weighed in last 3 months" \ "" \ "Times received Vitamin A (Ages 6 mos. - 2 yrs.)" \ "" \
			"Enrolled in school (Ages 7-12)" \ "" \ "Enrolled in school (Ages 13-15)" \ "" \ "\% school attendance last 2 weeks (Ages 7-12)" \ "" \
			"\% school attendance last 2 weeks (Ages 13-15)" \ "" \ ">85\% attendance last 2 weeks (Ages 7-12)" \ "" \ ">85\% attendance last 2 weeks (Ages 13-15)")
	ctitles("", "Observations", "Control Mean", "Treatment Mean");


frmttable, statmat(NC_b_se) 
	substat(1) 
	sdec(3) 
	ctitles("Treatment Effect (No Controls)") 
	merge;

//final table
#delimit ;

frmttable using "appendix_table6", 
	statmat(FE_b_se) 
	substat(1) 
	sdec(3) 
	ctitles("Treatment Effect (District FE)")
	merge 
	tex 
	fragment 
	replace;

#delimit cr






