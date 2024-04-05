

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
		global PKH = "$PKH/"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local latex = "`output'/latex"
local TARGET2 = "$TARGET2"

cd "`PKH'"

//Set working directory:
cd "`pkhdata'"

*** 3. CODING INFANT MORTALITY BY WAVE ***

*Wave I
******************************************************************************************
//Creating mother level information on date of survey for use in determining children's ages
******************************************************************************************
use "Wave I/Data/I_fin", clear

//Generate survey dates using first survey attempt with at least partial completion
gen survey_date=mdy(iivw_mm_1, iivw_dd_1, iivw_yy_1) if ihkw_1==1 | ihkw_1==2
replace survey_date=mdy(iivw_mm_2, iivw_dd_2, iivw_yy_2) if survey_date==. & ihkw_2==1 | ihkw_2==2
replace survey_date=mdy(iivw_mm_3, iivw_dd_3, iivw_yy_3) if survey_date==. & ihkw_3==1 | ihkw_3==2

gen survey_mon = ym(iivw_yy_1, iivw_mm_1) if ihkw_1==1 | ihkw_1==2
replace survey_mon= ym(iivw_yy_2, iivw_mm_2) if survey_mon==. & ihkw_2==1 | ihkw_2==2
replace survey_mon= ym(iivw_yy_3, iivw_mm_3) if survey_mon==. & ihkw_3==1 | ihkw_3==2

gen survey_year = iivw_yy_1 if ihkw_1 == 1 | ihkw_1 == 2
replace survey_year = iivw_yy_2 if survey_year == . & ihkw_2 == 1 | ihkw_2 == 2
replace survey_year = iivw_yy_3 if survey_year == . & ihkw_3 == 1 | ihkw_3 == 2

gen survey_month_specif = iivw_mm_1 if ihkw_1 == 1 | ihkw_1 == 2
replace survey_month_specif = iivw_mm_2 if survey_year == . & ihkw_2 == 1 | ihkw_2 == 2
replace survey_month_specif = iivw_mm_3 if survey_year == . & ihkw_3 == 1 | ihkw_3 == 2

tempfile survey_date_info_w1
save `survey_date_info_w1'

merge 1:m iid using "Wave I/Data/I_CH_fin"

tab _merge
tab ich01 _merge, m
drop if _merge == 1
drop _merge 

codebook survey_date

sort iid
tempfile pregnancy_last24months_w1
save `pregnancy_last24months_w1'


keep iid survey_date survey_month
duplicates drop
tempfile pregnancy_last24months_id_w1
save `pregnancy_last24months_id_w1'


*****************************************************************
//Info on pregnancies for children alive and living in house
*****************************************************************

use "Wave I/Data/R_fin", clear

//Generate survey dates using first survey attempt with at least partial completion
gen survey_date=mdy(rivw_mm_1, rivw_dd_1, rivw_yy_1) if rhkw_1==1 | rhkw_1==2
replace survey_date=mdy(rivw_mm_2, rivw_dd_2, rivw_yy_2) if survey_date==. & rhkw_2==1 | rhkw_2==2
replace survey_date=mdy(rivw_mm_3, rivw_dd_3, rivw_yy_3) if survey_date==. & rhkw_3==1 | rhkw_3==2

gen survey_month = ym(rivw_yy_1, rivw_mm_1) if rhkw_1==1 | rhkw_1==2
replace survey_month= ym(rivw_yy_2, rivw_mm_2) if survey_month==. & rhkw_2==1 | rhkw_2==2
replace survey_month= ym(rivw_yy_3, rivw_mm_3) if survey_month==. & rhkw_3==1 | rhkw_3==2

keep rid survey_date survey_month
sort rid

merge 1:m rid using "Wave I/Data/R_AR_01_fin"
tab _merge 

//drop if don't have info on mother's number (dead, not in house or missing)
drop if rar06 >= 51
tostring rar06, gen(mother_id) format(%02.0f)

gen iid = substr(rid,1,3) + "2" + substr(rid,5,5) + mother_id
keep if rar04_yy >= 2006 | (rar04_yy == 2005 & rar04_mm >= 6 & rar04_mm < .)
gen born_2005 = (rar04_yy == 2005)


**Survey was between Jun-Aug 2007, so if born in 2007, then within 6-8 months; Jan 2006 within 17-19 months, so good for within 24 and 21, 
gen live_in_house_18 = 1 if birth_diff <= 547.875
replace live_in_house_18 = 0 if birth_diff > 547.875 & birth_diff < .
replace live_in_house_18 = 1 if rar04_yy == 2007

gen live_in_house_over18 		= 1 if birth_diff > 547.875 & birth_diff < .
replace live_in_house_over18 	= 0 if birth_diff <= 547.875
replace live_in_house_over18 = 0 if rar04_yy == 2007 

gen live_in_house_12to24 = 1 if birth_diff > 365.25 & birth_diff <= 730.50
replace live_in_house_12to24 = 0 if birth_diff <= 365.25 | (birth_diff > 730.50 & birth_diff < .)
replace live_in_house_12to24 = 0 if rar04_yy == 2007

gen live_in_house_24 = 1 if birth_diff <= 730.50
replace live_in_house_24 = 0 if birth_diff > 730.50 & birth_diff < .
replace live_in_house_24 = 1 if rar04_yy == 2007 | rar04_yy == 2006

gen live_in_house_21 = 1 if birth_diff <= 639.1875 
replace live_in_house_21 = 0 if birth_diff > 639.1875 & birth_diff < .
replace live_in_house_21 = 1 if rar04_yy == 2007 | rar04_yy == 2006

gen live_in_house_12to21 = 1 if birth_diff > 365.25 & birth_diff <= 639.1875
replace live_in_house_12to21 = 0 if birth_diff <= 365.25 | (birth_diff > 639.1875 & birth_diff < .)
replace live_in_house_12to21 = 0 if rar04_yy == 2007

gen live_in_house = 1


//Merging with mother dataset on births in last 24 months
sort iid
merge m:1 iid using `pregnancy_last24months_id_w1', generate(_mx)
tab _mx

tab live_in_house _mx, m
tab live_in_house_18 _mx , m
tab live_in_house_over18 _mx, m
tab live_in_house_12to24 _mx, m
tab live_in_house_24 _mx, m
tab live_in_house_21 _mx, m
tab live_in_house_12to21 _mx, m

//Keeping only if matching with mother surveyed from I_CH
keep if _mx == 3
drop _mx

gen live_in_bdiff_miss = 1 if birth_diff == . & live_in_house_18 != 1

tempfile kids_input_w1
save `kids_input_w1'

//Saving mother level dataset with total number of kids under 18 living in house
bys iid: egen live_in_house_18_tot = total(live_in_house_18)
bys iid: egen live_in_house_over18_tot = total(live_in_house_over18)
bys iid: egen live_in_house_tot = total(live_in_house)
bys iid: egen live_in_bdiff_miss_tot = total(live_in_bdiff_miss)

gen preg_type = 4
keep live_in_house_18_tot live_in_bdiff_miss_tot live_in_house_tot iid survey_month survey_date live_in_house_over18_tot 
duplicates drop

sort iid
tempfile live_in_house_w1
save `live_in_house_w1'

//Creating kid level information from household roster for direct merge later
use `kids_input_w1', clear
tostring(rar00), gen(kid) format(%02.0f)
gen kid_id = substr(rid,1,3) + "4" + substr(rid,5,5) + kid
duplicates drop

rename live_in_house_18 live_in_house_18_ind
rename live_in_house_over18 live_in_house_over18_ind 
rename live_in_house_12to24 live_in_house_12to24_ind
rename live_in_house_24 live_in_house_24_ind
rename live_in_house_21 live_in_house_21_ind
rename live_in_house_12to21 live_in_house_12to21_ind

keep kid_id live_in_house_18_ind live_in_house_over18_ind live_in_house_12to24_ind live_in_house_24 live_in_house_21 live_in_house_12to21 live_in_house birth_mon birth_diff born_2005 
sort kid_id

//Saving kid level dataset with info on under 18 and living in house
tempfile kids_in_house_w1
save `kids_in_house_w1'

*****************************************************************
//Info on pregnancies for children alive but not living in house
*****************************************************************

use "Wave I/Data/R_fin", clear
//Generate survey dates using first survey attempt with at least partial completion
gen survey_date=mdy(rivw_mm_1, rivw_dd_1, rivw_yy_1) if rhkw_1==1 | rhkw_1==2
replace survey_date=mdy(rivw_mm_2, rivw_dd_2, rivw_yy_2) if survey_date==. & rhkw_2==1 | rhkw_2==2
replace survey_date=mdy(rivw_mm_3, rivw_dd_3, rivw_yy_3) if survey_date==. & rhkw_3==1 | rhkw_3==2

gen survey_month = ym(rivw_yy_1, rivw_mm_1) if rhkw_1==1 | rhkw_1==2
replace survey_month= ym(rivw_yy_2, rivw_mm_2) if survey_month==. & rhkw_2==1 | rhkw_2==2
replace survey_month= ym(rivw_yy_3, rivw_mm_3) if survey_month==. & rhkw_3==1 | rhkw_3==2

keep rid survey_date survey_month
sort rid

merge 1:m rid using "Wave I/Data/R_AR_26_fin"
tab _merge 
keep if _merge == 3
drop _merge 

//drop if don't have info on mother's number (dead, out of household, missing)
drop if rar32 >= 51
tostring rar32, gen(mother_id) format(%02.0f)

gen iid = substr(rid,1,3) + "2" + substr(rid,5,5) + mother_id
keep if rar30_yy >= 2006 | (rar30_yy == 2005 & rar30_mm >= 6 & rar30_mm < .)
gen born_2005 = (rar30_yy == 2005)


**Survey between Jun-Aug 2007, so if born in 2007 then within 10-12 months
gen live_out_house_18 = 1 if birth_diff <= 547.875
replace live_out_house_18 = 0 if birth_diff > 547.875 & birth_diff < .
replace live_out_house_18 = 1 if rar30_yy == 2007

gen live_out_house_over18 = 1 if birth_diff > 547.875 & birth_diff < .
replace live_out_house_over18 = 0 if birth_diff <= 547.875 
replace live_out_house_over18 = 0 if rar30_yy == 2007

gen live_out_house_12to24 = 1 if birth_diff > 365.25 & birth_diff <= 730.50
replace live_out_house_12to24 = 0 if birth_diff <= 365.25 | (birth_diff > 730.50 & birth_diff < .)
replace live_out_house_12to24 = 0 if rar30_yy == 2007

gen live_out_house_24 = 1 if birth_diff <= 730.50
replace live_out_house_24 = 0 if birth_diff > 730.50 & birth_diff < .
replace live_out_house_24 = 1 if rar30_yy == 2007 | rar30_yy == 2006

gen live_out_house_21 = 1 if birth_diff <= 639.1875 
replace live_out_house_21 = 0 if birth_diff > 639.1875 & birth_diff < .
replace live_out_house_21 = 1 if rar30_yy == 2007 | rar30_yy == 2006

gen live_out_house_12to21 = 1 if birth_diff > 365.25 & birth_diff <= 639.1875
replace live_out_house_12to21 = 0 if birth_diff <= 365.25 | (birth_diff > 639.1875 & birth_diff < .)
replace live_out_house_12to21 = 0 if rar30_yy == 2007

gen live_out_house = 1


//Merging with mother dataset on births in last 24 months
sort iid
merge m:1 iid using `pregnancy_last24months_id_w1', generate(_mx)
tab _mx

tab live_out_house _mx, m
tab live_out_house_18 _mx , m
tab live_out_house_12to24 _mx, m
tab live_out_house_21 _mx, m
tab live_out_house_24 _mx, m
tab live_out_house_12to21 _mx, m

//Keeping only if matching with mother surveyed from I_CH
keep if _mx == 3
drop _mx

gen live_out_bdiff_miss = 1 if birth_diff == . & live_out_house_18 != 1

tempfile kids_out_input_w1
save `kids_out_input_w1'

bys iid: egen live_out_house_18_tot = total(live_out_house_18)
bys iid: egen live_out_house_over18_tot = total(live_out_house_over18)
bys iid: egen live_out_house_tot = total(live_out_house)
bys iid: egen live_out_bdiff_miss_tot = total(live_out_bdiff_miss)

gen preg_type = 4
keep live_out_house_18_tot live_out_bdiff_miss_tot live_out_house_tot iid live_out_house_over18_tot
duplicates drop

//Saving mother level dataset with total number of kids under 18 living outside house
sort iid
tempfile live_out_house_w1
save `live_out_house_w1'


//Creating child level dataset
use `kids_out_input_w1', clear
//51 is used to signify those who live outside of household for individuals
//tostring(rar00), gen(kid) format(%02.0f)
gen kid_id = substr(rid,1,3) + "4" + substr(rid,5,5) + "51"
duplicates drop
duplicates report kid_id

rename live_out_house_18 live_out_house_18_ind
rename live_out_house_over18 live_out_house_over18_ind 
rename live_out_house_12to24 live_out_house_12to24_ind
rename live_out_house_24 live_out_house_24_ind
rename live_out_house_21 live_out_house_21_ind
rename live_out_house_12to21 live_out_house_12to21_ind

keep kid_id live_out_house_18_ind live_out_house_over18_ind live_out_house_12to24_ind live_out_house_24_ind live_out_house_21_ind live_out_house_12to21_ind live_out_house birth_mon birth_diff born_2005 rar30_mm rar30_yy
sort kid_id

//Saving kid level dataset with info on under 18 but not living in house
tempfile kids_out_house_w1
save `kids_out_house_w1', replace


***************************************************************************
//Get info on kids born and remained alive from I_CH 
***************************************************************************
use "Wave I/Data/I_CH_fin", clear
rename iid id

//Keeping only live births
keep if ich02 == 4

//Generating the child level id number for merges using information from the household rosters
tostring(ich03_cd), gen(kid) format(%02.0f)
gen kid_id = substr(id,1,3) + "4" + substr(id,5,5) + kid
duplicates tag kid_id, gen(tagx)

//Dropping births alive and then died 
drop if ich03_cd == 52


sort kid_id
merge m:1 kid_id using `kids_in_house_w1', generate(kid_in_house_merge)
tab kid_in_house_merge

tab live_in_house_18_ind kid_in_house_merge, m
tab live_in_house_12to24_ind kid_in_house_merge, m
tab live_in_house_21_ind kid_in_house_merge, m
tab live_in_house_24_ind kid_in_house_merge, m
tab live_in_house_12to21_ind kid_in_house_merge, m

tab kid_id if live_in_house_18_ind == 1 & kid_in_house_merge == 2
tab kid_id if live_in_house_12to24_ind == 1 & kid_in_house_merge == 2
tab kid_id if live_in_house_21_ind == 1 & kid_in_house_merge == 2
tab kid_id if live_in_house_24_ind == 1 & kid_in_house_merge == 2
tab kid_id if live_in_house_12to21_ind == 1 & kid_in_house_merge == 2

tab live_in_house_18_ind live_in_house_over18_ind, m

drop if kid_in_house_merge == 2

sort kid_id
merge m:1 kid_id using `kids_out_house_w1', generate(kid_out_house_merge)
tab kid_out_house_merge

tab live_out_house_18_ind kid_out_house, m
tab live_out_house_12to24_ind kid_out_house, m
tab live_out_house_21_ind kid_out_house, m
tab live_out_house_24_ind kid_out_house, m
tab live_out_house_12to21_ind kid_out_house, m

tab kid_id if live_out_house_18_ind == 1 & kid_out_house_merge == 2
tab kid_id if live_out_house_12to24_ind == 1 & kid_out_house_merge == 2
tab kid_id if live_out_house_21_ind == 1 & kid_out_house_merge == 2
tab kid_id if live_out_house_24_ind == 1 & kid_out_house_merge == 2
tab kid_id if live_out_house_12to21_ind == 1 & kid_out_house_merge == 2

tab kid_out_house if live_out_house_18_ind == 1 | live_out_house_12to24_ind == 1, m
tab live_out_house_18_ind live_out_house_over18_ind, m

drop if kid_out_house_merge == 2


//For alive and in house
gen within_18 = 1 if live_in_house_18_ind == 1 & ich03_cd < 51
replace within_18 = 0 if live_in_house_over18_ind == 1 & ich03_cd < 51

gen between_12to24 = 1 if live_in_house_12to24_ind == 1 & ich03_cd < 51
gen within_24 = 1 if live_in_house_24_ind == 1 & ich03_cd < 51
gen within_21 = 1 if live_in_house_21_ind == 1 & ich03_cd < 51
gen between_12to21 = 1 if live_in_house_12to21_ind == 1 & ich03_cd < 51

//For alive and out of house
replace within_18 = 1 if live_out_house_18_ind == 1 & ich03_cd == 51
replace within_18 = 0 if live_out_house_over18_ind == 1 & ich03_cd == 51

replace between_12to24 = 1 if live_out_house_12to24_ind == 1 & ich03_cd == 51
replace within_24 = 1 if live_out_house_24_ind == 1 & ich03_cd == 51
replace within_21 = 1 if live_out_house_21_ind == 1 & ich03_cd == 51
replace between_12to21 = 1 if live_out_house_12to21_ind == 1 & ich03_cd == 51

rename id  mother_id
gen id = kid_id

**If mother_id_ind == 0 then "id" is kid's id, if mother_id_ind == 1 then "id" is mother's id 
gen mother_id_ind = 0
gen dead = 0
gen dead_0to28_days = 0
gen dead_0to12_mons = 0
gen dead_1to12_mons = 0
gen within_24_old_version = 1

 
drop if within_24 == .
tempfile kids_alive_w1
save `kids_alive_w1'

*********************************************************
//Get info on children born live but then died from I_RH
*********************************************************

use "Wave I/Data/I_RH13_fin", clear
sort iid

**The survey dates are between June-Aug 2007, so any children born before June 2005 will be beyond 24 months
keep if irh13_yy >= 2006 | (irh13_yy == 2005 & irh13_mm >= 6 & irh13_mm < .)

//Drop empty observations
drop if irh13_yy == .
drop if irh13_yy == .y 

sort iid
//Merge info on survey date
merge m:1 iid using `survey_date_info_w1'
tab _merge
keep if _merge == 3


//Generate birth month years
gen birth_mon = ym(irh13_yy, irh13_mm)
gen birth_year = irh13_yy


gen elapsed_mon = survey_mon - birth_mon
gen death_elapsed_diff = age_died_mon - elapsed_mon 
tab death_elapsed, m


//Dummy for whether okay to keep this obs or not, based on age/date info
tab elapsed_mon age_died_mon if  death_elapsed_diff == 1, m
gen elapsed_death_okay = 1 if death_elapsed_diff <= 1


**the 2007 obs must be within 8 months of survey (b/c took place June-Aug 2007)
tab irh13_mm irh13_yy if elapsed_death_okay ==., m
tab age_died_days age_died_mon if irh13_yy == 2007 & irh13_mm == .y & elapsed_death_okay == .
replace elapsed_mon = 8 if irh13_yy == 2007 & irh13_mm == .y & elapsed_death_okay == .
replace age_died_mon = 8 if irh13_yy == 2007 & age_died_mon == 72 & irh13_mm == .y & elapsed_death_okay == .
replace age_died_days = 243.5 if irh13_yy == 2007 & age_died_mon == 72 & irh13_mm == .y & elapsed_death_okay == .
replace elapsed_death_okay = 1 if irh13_yy == 2007 & irh13_mm == .y & elapsed_death_okay == .

**the 2006 obs are most likely within 18 months of the survey (range from Dec 2005 - Feb 2006 or after to be included)
**definitely fall within 24 months, include in both to err on side of conservatism
replace elapsed_mon = 17 if irh13_yy == 2006 & irh13_mm == .y & elapsed_death_okay == .
replace elapsed_death_okay =1 if irh13_yy == 2006 & irh13_mm == .y & elapsed_death_okay == .

**Correct 2006 obs which have age of death greater than time elapsed since birth
replace elapsed_death_okay = 1 if irh13_yy == 2006 & elapsed_mon == 18 & age_died_mon == 24
replace age_died_days = 60.875 if irh13_yy == 2006 & elapsed_mon == 18 & age_died_mon == 24
replace age_died_mon = 2 if irh13_yy == 2006 & elapsed_mon == 18 & age_died_mon == 24

replace elapsed_death_okay = 1 if irh13_yy == 2006 & elapsed_mon == 14 & age_died_mon == 48
replace age_died_days = 121.75 if irh13_yy == 2006 & elapsed_mon == 14 & age_died_mon == 48
replace age_died_mon = 4 if irh13_yy == 2006 & elapsed_mon == 14 & age_died_mon == 48

replace elapsed_death_okay = 1 if irh13_yy == 2006 & elapsed_mon == 15 & age_died_mon == 30
replace age_died_days = 456.5625 if irh13_yy == 2006 & elapsed_mon == 15 & age_died_mon == 30
replace age_died_mon = 15 if irh13_yy == 2006 & elapsed_mon == 15 & age_died_mon == 30

**Correct 2005 obs which have age of death greater than time elapsed since birth
replace elapsed_death_okay = 1 if irh13_yy == 2005 & elapsed_mon == 18 & age_died_mon == 36
replace age_died_days = 91.3125 if irh13_yy == 2005 & elapsed_mon == 18 & age_died_mon == 36
replace age_died_mon = 3 if irh13_yy == 2005 & elapsed_mon == 18 & age_died_mon == 36

replace elapsed_death_okay = 1 if irh13_yy == 2005 & elapsed_mon == 22 & age_died_mon == 27
replace age_died_days = 669.625 if irh13_yy == 2005 & elapsed_mon == 22 & age_died_mon == 27
replace age_died_mon = 22 if irh13_yy == 2005 & elapsed_mon == 22 & age_died_mon == 27

**Manual mop-up of specific observations with clear fixes
replace age_died_mon = 3 if iid == "36820090202" & elapsed_death_okay == .
replace age_died_days = 91.3125 if iid =="36820090202" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "36820090202" & elapsed_death_okay == .

replace age_died_mon = 2 if iid == "47320020405" & elapsed_death_okay == .
replace age_died_days = 60.875 if iid == "47320020405" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "47320020405" & elapsed_death_okay == .

replace age_died_mon = 4 if iid == "47320020602" & elapsed_death_okay == .
replace age_died_days = 121.75 if iid == "47320020602" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "47320020602" & elapsed_death_okay == .

replace age_died_mon = 4 if iid == "48220080202" & elapsed_death_okay == .
replace age_died_days = 121.75 if iid == "48220080202" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "48220080202" & elapsed_death_okay == .

replace elapsed_mon = 23 if iid == "51520060302" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "51520060302" & elapsed_death_okay == .

replace age_died_mon = 4 if iid == "53620060205" & elapsed_death_okay == .
replace age_died_days = 121.75 if iid == "53620060205" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "53620060205" & elapsed_death_okay == .

replace age_died_mon = 6 if iid == "65720050402" & elapsed_death_okay == .
replace age_died_days = 182.625 if iid == "65720050402" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "65720050402" & elapsed_death_okay == .

replace age_died_mon = 8 if iid == "44120030202" & elapsed_death_okay == .
replace age_died_days = 243.5 if iid == "44120030202" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "44120030202" & elapsed_death_okay == .

replace age_died_mon = 10 if iid == "32820140402" & elapsed_death_okay == .
replace age_died_days = 304.375 if iid == "32820140402" & elapsed_death_okay == .
replace elapsed_death_okay = 1 if iid == "32820140402" & elapsed_death_okay == .

**the 2005 obs can't be within 18 months of the survey (need to be April, 2008 or after to be within 18 months)
**and most likely is not within 24 months of the survey (ambiguous anyway and would need to be June-Aug 2005)
replace elapsed_death_okay = 0 if irh13_yy == 2005 & elapsed_mon == .
replace elapsed_death_okay = 1 if marker == 1
replace elapsed_death_okay = 1 if age_died_mon > 12 & elapsed_mon> 24 & age_died_mon < . & elapsed_mon < .
replace elapsed_death_okay = 1 if (age_died_mon >= 1 & age_died_mon < 12) & (elapsed_mon >= 1 & elapsed_mon <= 18)


tab elapsed_death_okay, m

tab death_elapsed if elapsed_death == 1, m
drop death_elapsed_diff
gen death_elapsed_diff = age_died_mon - elapsed_mon
tab death_elapsed if elapsed_death == 1, m

tab elapsed_mon age_died_mon if death_elapsed > 0, m

drop if elapsed_death_okay == 0
drop marker elapsed_death_okay death_elapsed_diff


//Generate dummies for whether mother/child id and if dead/alive
gen mother_id_ind = 1
gen dead = 1


gen dead_0to28_days = 1 if age_died_days <= 28
replace dead_0to28_days = 0 if age_died_days > 28 & age_died_days < .

gen dead_0to12_mons = 1 if age_died_mon <= 12
replace dead_0to12_mons = 0 if age_died_mon > 12 & age_died_mon < .

gen dead_1to12_mons = 1 if age_died_mon >= 1 & age_died_mon <= 12
replace dead_1to12_mons = 0 if (age_died_mon < 1 |(age_died_mon > 12 & age_died_mon < .))

gen within_18 = 1 if elapsed_mon <= 18
replace within_18 = 0 if elapsed_mon > 18 & elapsed_mon < .
gen within_24 = 1 if elapsed_mon <= 24
replace within_24 = 0 if elapsed_mon > 24 & elapsed_mon < .
gen within_21 = 1 if elapsed_mon <= 21
replace within_21 = 0 if elapsed_mon > 21 & elapsed_mon < .


gen between_12to24 = 1 if elapsed_mon > 12 & elapsed_mon <= 24
replace between_12to24 = 0 if elapsed_mon <=12 | (elapsed_mon > 24 & elapsed_mon < .)
gen between_12to21 = 1 if elapsed_mon > 12 & elapsed_mon <= 21
replace between_12to21 = 0 if elapsed_mon <= 12 | (elapsed_mon > 21 & elapsed_mon < .)

gen born_2005 = (irh13_yy == 2005)
keep iid survey_mon birth_mon birth_year dead* mother_id age_died_mon age_died_days elapsed_mon survey_year within* between_12to24 between_12to21 born_2005
rename iid id

append using `kids_alive_w1'

gen dead_0to28_days_last18 = dead_0to28_days if within_18 == 1
gen dead_0to28_days_last24 = dead_0to28_days if within_24 == 1

gen dead_0to12_mons_last24 = dead_0to12_mons if within_24 == 1
gen dead_0to12_mons_12to24 = dead_0to12_mons if between_12to24 == 1

gen dead_1to12_mons_last24 = dead_1to12_mons if within_24 == 1

gen dead_0to12_mons_last21 = dead_0to12_mons if within_21 == 1
gen dead_0to12_mons_12to21 = dead_0to12_mons if between_12to21 == 1

gen survey_round = 0

gen rid = substr(id,1,3) + "1" + substr(id,5,5)
sort rid 

tempfile infant_mortality_w1
save `infant_mortality_w1', replace 




*Wave III
*****************************************************************************************
//Creating mother level information on date of survey for use in determining children's ages
******************************************************************************************
use "Wave III/Data/i_fin", clear
duplicates drop

//Generate survey dates
gen survey_start_day = substr(istart_d,1,2)
destring survey_start_day, replace 
gen survey_start_mon = substr(istart_d,4,2)
destring survey_start_mon, replace 
gen survey_start_year = substr(istart_d,7,2)
destring survey_start_year, replace 
replace survey_start_year = survey_start_year + 2000

gen survey_finish_day = substr(ifinish_d,1,2)
destring survey_finish_day, replace 
gen survey_finish_mon = substr(ifinish_d,4,2)
destring survey_finish_mon, replace 
gen survey_finish_year = substr(ifinish_d,7,2)
destring survey_finish_year, replace 
replace survey_finish_year = survey_finish_year + 2000

gen survey_start_date=	mdy(survey_start_mon, survey_start_day, survey_start_year) 
gen survey_finish_date=	mdy(survey_finish_mon, survey_finish_day, survey_finish_year)
gen survey_days_elapse = survey_finish_date - survey_start_date

summ survey_days_elapse, d

gen survey_date = survey_start_date
gen survey_month = ym(survey_start_year, survey_start_mon) 
gen survey_year = survey_start_year

gen survey_m_specif = survey_start_mon

drop survey_start* survey_finish*


tempfile survey_date_info_w3
save `survey_date_info_w3'


merge 1:m iid using "Wave III/Data/I_CH_fin"
tab _merge 
tab ich01 _m, m
drop if _m == 1

//If ich03_cd is . means is not in household, still born, miscarriage, or live then died
sort iid
tempfile pregnancy_last24months_w3
save `pregnancy_last24months_w3'


keep iid survey_date survey_month
duplicates drop
tempfile pregnancy_last24months_id_w3
save `pregnancy_last24months_id_w3'


use "Wave III/Data/2_R_AR_01_fin", clear
duplicates drop
tempfile R_AR_01_w3_duplicates_drop
save `R_AR_01_w3_duplicates_drop'

*****************************************************************
//Info on pregnancies for children alive and living in house
*****************************************************************

use "Wave III/Data/R_fin", clear

duplicates drop

//Generate survey dates
gen survey_start_day = substr(rstart_d,1,2)
destring survey_start_day, replace 
gen survey_start_mon = substr(rstart_d,4,2)
destring survey_start_mon, replace 
gen survey_start_year = substr(rstart_d,7,2)
destring survey_start_year, replace 
replace survey_start_year = survey_start_year + 2000

gen survey_finish_day = substr(rfinish_d,1,2)
destring survey_finish_day, replace 
gen survey_finish_mon = substr(rfinish_d,4,2)
destring survey_finish_mon, replace 
gen survey_finish_year = substr(rfinish_d,7,2)
destring survey_finish_year, replace 
replace survey_finish_year = survey_finish_year + 2000

gen survey_start_date=	mdy(survey_start_mon, survey_start_day, survey_start_year) 
gen survey_finish_date=	mdy(survey_finish_mon, survey_finish_day, survey_finish_year)
gen survey_days_elapse = survey_finish_date - survey_start_date

summ survey_days_elapse, d

gen survey_date = survey_start_date
gen survey_month = ym(survey_start_year, survey_start_mon) 

drop survey_start* survey_finish*

duplicates drop rid, force

keep rid survey_date survey_month rar00x
sort rid

tempfile household_survey_dates_w3
save `household_survey_dates_w3'

merge 1:m rid using `R_AR_01_w3_duplicates_drop'
tab _m

//drop if don't have info on mother's number (dead, not in house or missing)
drop if rar06 >= 51
tostring rar06, gen(mother_id) format(%02.0f)

gen iid = substr(rid,1,3) + "2" + substr(rid,5,9) + mother_id

keep if rar04_yy >= 2008 | (rar04_yy == 2007 & rar04_mm >= 10 & rar04_mm < .)

gen born_2007 = (rar04_yy == 2007)


**Survey was between Oct-Dec 2009, so if born in 2009, then within 10-12 months; if born in Jan 2008, within 21-23 months, so 2008 obs are def within 24
gen live_in_house_18 = 1 if birth_diff <= 547.875
replace live_in_house_18 = 0 if birth_diff > 547.875 & birth_diff < .
replace live_in_house_18 = 1 if rar04_yy == 2009

gen live_in_house_over18 		= 1 if birth_diff > 547.875 & birth_diff < .
replace live_in_house_over18 	= 0 if birth_diff <= 547.875
replace live_in_house_over18 = 0 if rar04_yy == 2009 

gen live_in_house_12to24 = 1 if birth_diff > 365.25 & birth_diff <= 730.50
replace live_in_house_12to24 = 0 if birth_diff <= 365.25 | (birth_diff > 730.50 & birth_diff < .)
replace live_in_house_12to24 = 0 if rar04_yy == 2009

gen live_in_house_24 = 1 if birth_diff <= 730.50
replace live_in_house_24 = 0 if birth_diff > 730.50 & birth_diff < .
replace live_in_house_24 = 1 if rar04_yy == 2009 | rar04_yy == 2008

gen live_in_house_21 = 1 if birth_diff <= 639.1875 
replace live_in_house_21 = 0 if birth_diff > 639.1875 & birth_diff < .
replace live_in_house_21 = 1 if rar04_yy == 2009

gen live_in_house_12to21 = 1 if birth_diff > 365.25 & birth_diff <= 639.1875
replace live_in_house_12to21 = 0 if birth_diff <= 365.25 | (birth_diff > 639.1875 & birth_diff < .)
replace live_in_house_12to21 = 0 if rar04_yy == 2009

gen live_in_house = 1


//Merging with mother dataset on births in last 24 months
sort iid
merge m:1 iid using `pregnancy_last24months_id_w3', generate(_mx)
tab _mx


//matching with mother surveyed from I_CH
keep if _mx == 3
drop _mx

gen live_in_bdiff_miss = 1 if birth_diff == . & live_in_house_18 != 1

tempfile kids_input_w3
save `kids_input_w3'

bys iid: egen live_in_house_18_tot = total(live_in_house_18)
bys iid: egen live_in_house_over18_tot = total(live_in_house_over18)
bys iid: egen live_in_house_tot = total(live_in_house)
bys iid: egen live_in_bdiff_miss_tot = total(live_in_bdiff_miss)

gen preg_type = 4
keep live_in_house_18_tot live_in_bdiff_miss_tot live_in_house_tot iid survey_date survey_month live_in_house_over18_tot 
duplicates drop

//Saving mother level dataset with total number of kids under 18 living in house
sort iid
tempfile live_in_house_w3
save `live_in_house_w3'

//Creating kid level information from household roster for direct merge later
use `kids_input_w3', clear
tostring(rar00), gen(kid) format(%02.0f)
gen kid_id = substr(rid,1,3) + "4" + substr(rid,5,9) + kid
duplicates drop


rename live_in_house_18 live_in_house_18_ind
rename live_in_house_over18 live_in_house_over18_ind 
rename live_in_house_12to24 live_in_house_12to24_ind
rename live_in_house_24 live_in_house_24_ind
rename live_in_house_21 live_in_house_21_ind
rename live_in_house_12to21 live_in_house_12to21_ind

keep kid_id live_in_house_18_ind live_in_house_over18_ind live_in_house_12to24_ind live_in_house_24 live_in_house_21 live_in_house_12to21 live_in_house birth_mon birth_diff born_2007
sort kid_id

//Saving kid level dataset with info on under 18 and living in house
tempfile kids_in_house_w3
save `kids_in_house_w3', replace 


*****************************************************************
//Info on pregnancies for children alive but not living in house
*****************************************************************

use `household_survey_dates_w3', clear

merge 1:m rid using "Wave III/Data/R_AR_26_fin"
tab _m
keep if _m == 3


//drop if don't have info on mother's number (dead, out of household, missing)
drop if rar32 >= 51
tostring rar32, gen(mother_id) format(%02.0f)

gen iid = substr(rid,1,3) + "2" + substr(rid,5,9) + mother_id

keep if rar30_yy >= 2008 | (rar30_yy == 2007 & rar30_mm >= 10 & rar30_mm < .)

gen born_2007 = (rar30_yy == 2007)


**Survey between Oct-Dec 2009, so if born in 2009 then within 10-12 months

gen live_out_house_18 = 1 if birth_diff <= 547.875
replace live_out_house_18 = 0 if birth_diff > 547.875 & birth_diff < .
replace live_out_house_18 = 1 if rar30_yy == 2009

gen live_out_house_over18 = 1 if birth_diff > 547.875 & birth_diff < .
replace live_out_house_over18 = 0 if birth_diff <= 547.875 
replace live_out_house_over18 = 0 if rar30_yy == 2009

gen live_out_house_12to24 = 1 if birth_diff > 365.25 & birth_diff <= 730.50
replace live_out_house_12to24 = 0 if birth_diff <= 365.25 | (birth_diff > 730.50 & birth_diff < .)
replace live_out_house_12to24 = 0 if rar30_yy == 2009

gen live_out_house_24 = 1 if birth_diff <= 730.50
replace live_out_house_24 = 0 if birth_diff > 730.50 & birth_diff < .
replace live_out_house_24 = 1 if rar30_yy == 2009 | rar30_yy == 2008

gen live_out_house_21 = 1 if birth_diff <= 639.1875 
replace live_out_house_21 = 0 if birth_diff > 639.1875 & birth_diff < .
replace live_out_house_21 = 1 if rar30_yy == 2009

gen live_out_house_12to21 = 1 if birth_diff > 365.25 & birth_diff <= 639.1875
replace live_out_house_12to21 = 0 if birth_diff <= 365.25 | (birth_diff > 639.1875 & birth_diff < .)
replace live_out_house_12to21 = 0 if rar30_yy == 2009

gen live_out_house = 1


//Merging with mother dataset on births in last 24 months
sort iid
merge m:1 iid using `pregnancy_last24months_id_w3', generate(_mx)
tab _mx

tab live_out_house _mx, m
tab live_out_house_18 _mx , m
tab live_out_house_12to24 _mx, m
tab live_out_house_21 _mx, m
tab live_out_house_24 _mx, m
tab live_out_house_12to21 _mx, m


//Keeping only if matching with mother surveyed from I_CH
keep if _mx == 3
drop _mx

gen live_out_bdiff_miss = 1 if birth_diff == . & live_out_house_18 != 1
tempfile kids_out_input_w3
save `kids_out_input_w3', replace

bys iid: egen live_out_house_18_tot = total(live_out_house_18)
bys iid: egen live_out_house_over18_tot = total(live_out_house_over18)
bys iid: egen live_out_house_tot = total(live_out_house)
bys iid: egen live_out_bdiff_miss_tot = total(live_out_bdiff_miss)

gen preg_type = 4
keep live_out_house_18_tot live_out_bdiff_miss_tot live_out_house_tot iid live_out_house_over18_tot
duplicates drop

//Saving mother level dataset with total number of kids under 18 living in house
sort iid
tempfile live_out_house_w3
save `live_out_house_w3'


//Creating child level dataset
use `kids_out_input_w3', clear
//51 is used to signify those who live outside of household for individuals
gen kid_id = substr(rid,1,3) + "4" + substr(rid,5,9) + "51"
duplicates drop

duplicates report kid_id
**No duplicates in terms of kid_id, so need not worry about when merge in later
rename live_out_house_18 live_out_house_18_ind
rename live_out_house_over18 live_out_house_over18_ind 
rename live_out_house_12to24 live_out_house_12to24_ind
rename live_out_house_24 live_out_house_24_ind
rename live_out_house_21 live_out_house_21_ind
rename live_out_house_12to21 live_out_house_12to21_ind

keep kid_id live_out_house_18_ind live_out_house_over18_ind live_out_house_12to24_ind live_out_house_24_ind live_out_house_21_ind live_out_house_12to21_ind live_out_house birth_mon birth_diff born_2007 rar30_mm rar30_yy
sort kid_id

//Saving kid level dataset with info on under 18 but not living in house
tempfile kids_out_house_w3
save `kids_out_house_w3', replace 



***************************************************************************
//Get info on kids born and remained alive from I_CH 
***************************************************************************
use "Wave III/Data/I_CH_fin", clear
rename iid id

//Keeping only live births
keep if ich02 == 4

//Generating the child level id number for merges using information from the household rosters
tostring(ich03_cd), gen(kid) format(%02.0f)
gen kid_id = substr(id,1,3) + "4" + substr(id,5,9) + kid
duplicates tag kid_id, gen(tagx)

//Dropping births alive and then died (tally elsewhere, where have age at death)
drop if ich03_cd >= 61 & ich03_cd <= 64
//Dropping stillbirths
drop if ich03_cd >= 71 & ich03_cd <.


sort kid_id
merge m:1 kid_id using `kids_in_house_w3', generate(kid_in_house_merge)
tab kid_in_house_merge

drop if kid_in_house_merge == 2



sort kid_id
merge m:1 kid_id using `kids_out_house_w3', generate(kid_out_house_merge) update 
tab kid_out_house_merge

drop if kid_out_house_merge == 2



//For alive and in house
gen within_18 = 1 if live_in_house_18_ind == 1 & ich03_cd < 51
replace within_18 = 0 if live_in_house_over18_ind == 1 & ich03_cd < 51

gen between_12to24 = 1 if live_in_house_12to24_ind == 1 & ich03_cd < 51
gen within_24 = 1 if live_in_house_24_ind == 1 & ich03_cd < 51
gen within_21 = 1 if live_in_house_21_ind == 1 & ich03_cd < 51
gen between_12to21 = 1 if live_in_house_12to21_ind == 1 & ich03_cd < 51

//For alive and out of house
replace within_18 = 1 if live_out_house_18_ind == 1 & ich03_cd == 51
replace within_18 = 0 if live_out_house_over18_ind == 1 & ich03_cd == 51

replace between_12to24 = 1 if live_out_house_12to24_ind == 1 & ich03_cd == 51
replace within_24 = 1 if live_out_house_24_ind == 1 & ich03_cd == 51
replace within_21 = 1 if live_out_house_21_ind == 1 & ich03_cd == 51
replace between_12to21 = 1 if live_out_house_12to21_ind == 1 & ich03_cd == 51

rename id  mother_id
gen id = kid_id

**If mother_id_ind == 0 then "id" is kid's id, if mother_id_ind == 1 then "id" is mother's id 
gen mother_id_ind = 0
gen dead = 0
gen dead_0to28_days = 0
gen dead_0to12_mons = 0
gen dead_1to12_mons = 0
gen within_24_old_version = 1



**We have 21 observations which are neither out of household nor dead, but not match with live in household
drop if within_24 == .

tempfile kids_alive_w3
save `kids_alive_w3', replace 

*********************************************************
//Get info on children born live but then died from I_RH
*********************************************************
use "Wave III/Data/I_RH13_fin", clear
sort iid

**The survey dates are between Oct-Dec 2009, so any children born before Oct. 2007 will be beyond 24 months
**We drop these observations for each of analysis below
keep if irh13_yy >= 2008 | (irh13_yy == 2007 & irh13_mm >= 10 & irh13_mm < .)
drop if irh13_yy >= .

sort iid
//Merge info on survey date
merge m:1 iid using `survey_date_info_w3'
tab _merge
keep if _merge == 3

gen elapsed_mon = survey_mon - birth_mon
gen death_elapsed_diff = age_died_mon - elapsed_mon
tab death_elapsed, m


//Dummy for whether okay to keep this obs or not, based on age/date info
gen elapsed_death_okay = 1 if death_elapsed_diff <= 0


tab irh13_mm irh13_yy if elapsed_death_okay ==., m
tab age_died_mon if irh13_yy == 2009 & elapsed_mon == .


replace elapsed_death_okay = 0 if irh13_yy == 2007 & elapsed_mon == .
replace elapsed_death_okay = 1 if irh13_yy == 2009 & elapsed_mon == .
replace elapsed_mon = 9 if irh13_yy == 2009 & elapsed_mon == .

tab age_died_days age_died_mon if irh13_yy == 2008 & elapsed_death_okay == ., m
replace elapsed_mon = 18 if irh13_yy == 2008 & elapsed_death_okay == . & age_died_days <= 10
replace elapsed_mon = 18 if irh13_yy == 2008 & elapsed_death_okay == . & age_died_mon == 6
replace elapsed_mon = 18 if irh13_yy == 2008 & elapsed_death_okay == . & age_died_mon == 2
replace elapsed_death_okay = 1 if irh13_yy == 2008 & elapsed_death_okay == .



drop death_elapsed_diff
gen death_elapsed_diff = age_died_mon - elapsed_mon
tab death_elapsed_diff, m


drop if elapsed_death_okay == 0
drop elapsed_death_okay death_elapsed_diff



//Generate dummies for whether mother/child id and if dead/alive
gen mother_id_ind = 1
gen dead = 1

gen dead_0to28_days = 1 if age_died_days <= 28
replace dead_0to28_days = 0 if age_died_days > 28 & age_died_days < .

gen dead_0to12_mons = 1 if age_died_mon <= 12
replace dead_0to12_mons = 0 if age_died_mon > 12 & age_died_mon < .

gen dead_1to12_mons = 1 if age_died_mon >= 1 & age_died_mon <= 12
replace dead_1to12_mons = 0 if (age_died_mon < 1 |(age_died_mon > 12 & age_died_mon < .))


gen within_18 = 1 if elapsed_mon <= 18
replace within_18 = 0 if elapsed_mon > 18 & elapsed_mon < .
gen within_24 = 1 if elapsed_mon <= 24
replace within_24 = 0 if elapsed_mon > 24 & elapsed_mon < .
gen within_21 = 1 if elapsed_mon <= 21
replace within_21 = 0 if elapsed_mon > 21 & elapsed_mon < .

gen between_12to24 = 1 if elapsed_mon > 12 & elapsed_mon <= 24
replace between_12to24 = 0 if elapsed_mon <=12 | (elapsed_mon > 24 & elapsed_mon < .)
gen between_12to21 = 1 if elapsed_mon > 12 & elapsed_mon <= 21
replace between_12to21 = 0 if elapsed_mon <= 12 | (elapsed_mon > 21 & elapsed_mon < .)

gen born_2007 = (irh13_yy == 2007)


keep iid survey_mon birth_mon birth_year dead* mother_id age_died_mon age_died_days elapsed_mon survey_year within* between_12to24 between_12to21 born_2007
rename iid id


append using `kids_alive_w3'


gen dead_0to28_days_last18 = dead_0to28_days if within_18 == 1
gen dead_0to28_days_last24 = dead_0to28_days if within_24 == 1

gen dead_0to12_mons_last24 = dead_0to12_mons if within_24 == 1
gen dead_1to12_mons_last24 = dead_1to12_mons if within_24 == 1
gen dead_0to12_mons_12to24 = dead_0to12_mons if between_12to24 == 1

gen dead_0to12_mons_last21 = dead_0to12_mons if within_21 == 1
gen dead_0to12_mons_12to21 = dead_0to12_mons if between_12to21 == 1


//Generate variable identifying household (absent of 4th digit specific to book) for future merges
gen rid_w3 = substr(id,1,3) + "1" + substr(id,5,9)
sort rid_w3
gen rid = substr(rid_w3,1,9)

gen survey_round = 1

tempfile infant_mortality_w3
save `infant_mortality_w3', replace



*Wave IV
*****************************************************************************************
//Creating mother level information on date of survey for use in determining children's ages
******************************************************************************************

use "Wave IV/I", clear
duplicates drop


//Generate survey dates
gen survey_start_day = substr(istart_d,1,2)
destring survey_start_day, replace 
gen survey_start_mon = substr(istart_d,4,2)
destring survey_start_mon, replace 
gen survey_start_year = substr(istart_d,7,2)
destring survey_start_year, replace 
replace survey_start_year = survey_start_year + 2000

gen survey_finish_day = substr(ifinish_d,1,2)
destring survey_finish_day, replace 
gen survey_finish_mon = substr(ifinish_d,4,2)
destring survey_finish_mon, replace 
gen survey_finish_year = substr(ifinish_d,7,2)
destring survey_finish_year, replace 
replace survey_finish_year = survey_finish_year + 2000

gen survey_start_date=	mdy(survey_start_mon, survey_start_day, survey_start_year) 
gen survey_finish_date=	mdy(survey_finish_mon, survey_finish_day, survey_finish_year)
gen survey_days_elapse = survey_finish_date - survey_start_date

summ survey_days_elapse, d

gen survey_date = survey_start_date
gen survey_month = ym(survey_start_year, survey_start_mon) 
gen survey_year = survey_start_year

gen survey_m_specif = survey_start_mon

drop survey_start* survey_finish*


tempfile survey_date_info_w4
save `survey_date_info_w4'


merge 1:m iid using "Wave IV/I_CH"
tab _merge 
tab ich01 _m, m
drop if _m == 1

//If ich03_cd is . means is not in household, still born, miscarriage, or live then died
sort iid
tempfile pregnancy_last24months_w4
save `pregnancy_last24months_w4'


keep iid survey_date survey_month
duplicates drop
tempfile pregnancy_last24months_id_w4
save `pregnancy_last24months_id_w4'


*****************************************************************
//Info on pregnancies for children alive and living in house
*****************************************************************

use "Wave IV/R", clear

duplicates drop

//Generate survey dates
gen survey_start_day = substr(rstart_d,1,2)
destring survey_start_day, replace 
gen survey_start_mon = substr(rstart_d,4,2)
destring survey_start_mon, replace 
gen survey_start_year = substr(rstart_d,7,2)
destring survey_start_year, replace 
replace survey_start_year = survey_start_year + 2000

gen survey_finish_day = substr(rfinish_d,1,2)
destring survey_finish_day, replace 
gen survey_finish_mon = substr(rfinish_d,4,2)
destring survey_finish_mon, replace 
gen survey_finish_year = substr(rfinish_d,7,2)
destring survey_finish_year, replace 
replace survey_finish_year = survey_finish_year + 2000

gen survey_start_date=	mdy(survey_start_mon, survey_start_day, survey_start_year) 
gen survey_finish_date=	mdy(survey_finish_mon, survey_finish_day, survey_finish_year)
gen survey_days_elapse = survey_finish_date - survey_start_date

summ survey_days_elapse, d

gen survey_date = survey_start_date
gen survey_month = ym(survey_start_year, survey_start_mon) 

drop survey_start* survey_finish*

duplicates drop rid, force

keep rid survey_date survey_month rar00x
sort rid

tempfile household_survey_dates_w4
save `household_survey_dates_w4'

merge 1:m rid using "Wave IV/2_R_AR_01"
tab _m


//drop if don't have info on mother's number (dead, not in house or missing)
drop if rar06 >= 51
tostring rar06, gen(mother_id) format(%02.0f)

gen iid = substr(rid,1,3) + "2" + substr(rid,5,13) + mother_id

keep if rar04_yy >= 2012 | (rar04_yy == 2011 & rar04_mm >= 9 & rar04_mm < .)

gen born_2011 = (rar04_yy == 2011)


**Survey was between Oct-Dec 2009, so if born in 2009, then within 10-12 months; if born in Jan 2008, within 21-23 months, so 2008 obs are def within 24, ambiguous on 18 and 21 months
gen live_in_house_18 = 1 if birth_diff <= 547.875
replace live_in_house_18 = 0 if birth_diff > 547.875 & birth_diff < .
replace live_in_house_18 = 1 if rar04_yy == 2013

gen live_in_house_over18 		= 1 if birth_diff > 547.875 & birth_diff < .
replace live_in_house_over18 	= 0 if birth_diff <= 547.875
replace live_in_house_over18 = 0 if rar04_yy == 2013

gen live_in_house_12to24 = 1 if birth_diff > 365.25 & birth_diff <= 730.50
replace live_in_house_12to24 = 0 if birth_diff <= 365.25 | (birth_diff > 730.50 & birth_diff < .)
replace live_in_house_12to24 = 0 if rar04_yy == 2013

gen live_in_house_24 = 1 if birth_diff <= 730.50
replace live_in_house_24 = 0 if birth_diff > 730.50 & birth_diff < .
replace live_in_house_24 = 1 if rar04_yy == 2013 | rar04_yy == 2012

gen live_in_house_21 = 1 if birth_diff <= 639.1875 
replace live_in_house_21 = 0 if birth_diff > 639.1875 & birth_diff < .
replace live_in_house_21 = 1 if rar04_yy == 2013

gen live_in_house_12to21 = 1 if birth_diff > 365.25 & birth_diff <= 639.1875
replace live_in_house_12to21 = 0 if birth_diff <= 365.25 | (birth_diff > 639.1875 & birth_diff < .)
replace live_in_house_12to21 = 0 if rar04_yy == 2013

gen live_in_house = 1


//Merging with mother dataset on births in last 24 months
sort iid
merge m:1 iid using `pregnancy_last24months_id_w4', generate(_mx)
tab _mx


//Keeping only if matching with mother surveyed from I_CH
keep if _mx == 3
drop _mx

gen live_in_bdiff_miss = 1 if birth_diff == . & live_in_house_18 != 1

tempfile kids_input_w4
save `kids_input_w4'

bys iid: egen live_in_house_18_tot = total(live_in_house_18)
bys iid: egen live_in_house_over18_tot = total(live_in_house_over18)
bys iid: egen live_in_house_tot = total(live_in_house)
bys iid: egen live_in_bdiff_miss_tot = total(live_in_bdiff_miss)

gen preg_type = 4
keep live_in_house_18_tot live_in_bdiff_miss_tot live_in_house_tot iid survey_date survey_month live_in_house_over18_tot 
duplicates drop

//Saving mother level dataset with total number of kids under 18 living in house
sort iid
tempfile live_in_house_w4
save `live_in_house_w4'


//Creating kid level information from household roster for direct merge later
use `kids_input_w4', clear
tostring(rar00), gen(kid) format(%02.0f)
gen kid_id = substr(rid,1,3) + "4" + substr(rid,5,13) + kid
duplicates drop


rename live_in_house_18 live_in_house_18_ind
rename live_in_house_over18 live_in_house_over18_ind 
rename live_in_house_12to24 live_in_house_12to24_ind
rename live_in_house_24 live_in_house_24_ind
rename live_in_house_21 live_in_house_21_ind
rename live_in_house_12to21 live_in_house_12to21_ind

keep kid_id live_in_house_18_ind live_in_house_over18_ind live_in_house_12to24_ind live_in_house_24 live_in_house_21 live_in_house_12to21 live_in_house birth_mon birth_diff born_2011
sort kid_id

//Saving kid level dataset with info on under 18 and living in house
tempfile kids_in_house_w4
save `kids_in_house_w4', replace 


*****************************************************************
//Info on pregnancies for children alive but not living in house
*****************************************************************

use `household_survey_dates_w4', clear

merge 1:m rid using "Wave IV/R_AR_26"
tab _m
keep if _m == 3


//drop if don't have info on mother's number (dead, out of household, missing)
drop if rar32 >= 51
tostring rar32, gen(mother_id) format(%02.0f)
gen iid = substr(rid,1,3) + "2" + substr(rid,5,13) + mother_id
keep if rar30_yy >= 2012 | (rar30_yy == 2011 & rar30_mm >= 9 & rar30_mm < .)

gen born_2011 = (rar30_yy == 2011)



**Survey between Oct-Dec 2009, so if born in 2009 then within 10-12 months

gen live_out_house_18 = 1 if birth_diff <= 547.875
replace live_out_house_18 = 0 if birth_diff > 547.875 & birth_diff < .
replace live_out_house_18 = 1 if rar30_yy == 2013

gen live_out_house_over18 = 1 if birth_diff > 547.875 & birth_diff < .
replace live_out_house_over18 = 0 if birth_diff <= 547.875 
replace live_out_house_over18 = 0 if rar30_yy == 2013

gen live_out_house_12to24 = 1 if birth_diff > 365.25 & birth_diff <= 730.50
replace live_out_house_12to24 = 0 if birth_diff <= 365.25 | (birth_diff > 730.50 & birth_diff < .)
replace live_out_house_12to24 = 0 if rar30_yy == 2013

gen live_out_house_24 = 1 if birth_diff <= 730.50
replace live_out_house_24 = 0 if birth_diff > 730.50 & birth_diff < .
replace live_out_house_24 = 1 if rar30_yy == 2013 | rar30_yy == 2012

gen live_out_house_21 = 1 if birth_diff <= 639.1875 
replace live_out_house_21 = 0 if birth_diff > 639.1875 & birth_diff < .
replace live_out_house_21 = 1 if rar30_yy == 2013

gen live_out_house_12to21 = 1 if birth_diff > 365.25 & birth_diff <= 639.1875
replace live_out_house_12to21 = 0 if birth_diff <= 365.25 | (birth_diff > 639.1875 & birth_diff < .)
replace live_out_house_12to21 = 0 if rar30_yy == 2013

gen live_out_house = 1


//Merging with mother dataset on births in last 24 months
sort iid
merge m:1 iid using `pregnancy_last24months_id_w4', generate(_mx)
tab _mx



//Keeping only if matching with mother surveyed from I_CH
keep if _mx == 3
drop _mx

gen live_out_bdiff_miss = 1 if birth_diff == . & live_out_house_18 != 1
tempfile kids_out_input_w4
save `kids_out_input_w4', replace

bys iid: egen live_out_house_18_tot = total(live_out_house_18)
bys iid: egen live_out_house_over18_tot = total(live_out_house_over18)
bys iid: egen live_out_house_tot = total(live_out_house)
bys iid: egen live_out_bdiff_miss_tot = total(live_out_bdiff_miss)

gen preg_type = 4
keep live_out_house_18_tot live_out_bdiff_miss_tot live_out_house_tot iid live_out_house_over18_tot
duplicates drop

//Saving mother level dataset with total number of kids under 18 living in house
sort iid
tempfile live_out_house_w4
save `live_out_house_w4'


//Creating child level dataset
use `kids_out_input_w4', clear
gen kid_id = substr(rid,1,3) + "4" + substr(rid,5,13) + "51"
duplicates drop

duplicates report kid_id

rename live_out_house_18 live_out_house_18_ind
rename live_out_house_over18 live_out_house_over18_ind 
rename live_out_house_12to24 live_out_house_12to24_ind
rename live_out_house_24 live_out_house_24_ind
rename live_out_house_21 live_out_house_21_ind
rename live_out_house_12to21 live_out_house_12to21_ind

keep kid_id live_out_house_18_ind live_out_house_over18_ind live_out_house_12to24_ind live_out_house_24_ind live_out_house_21_ind live_out_house_12to21_ind live_out_house birth_mon birth_diff born_2011 rar30_mm rar30_yy
sort kid_id

//Saving kid level dataset with info on under 18 but not living in house
tempfile kids_out_house_w4
save `kids_out_house_w4', replace 



***************************************************************************
//Get info on kids born and remained alive from I_CH 
***************************************************************************

use "Wave IV/I_CH", clear
rename iid id

//Keeping only live births
keep if ich02 == 4

//Generating the child level id number for merges using information from the household rosters
destring ich03_cd, replace 
tostring(ich03_cd), gen(kid) format(%02.0f)
gen kid_id = substr(id,1,3) + "4" + substr(id,5,13) + kid
duplicates tag kid_id, gen(tagx)

//Dropping births alive and then died (tally elsewhere, where have age at death)
drop if ich03_cd >= 61 & ich03_cd <= 64
//Dropping stillbirths
drop if ich03_cd >= 71 & ich03_cd <.


//Merging using datasets created from marriedwomen16to49years do file
sort kid_id
merge m:1 kid_id using `kids_in_house_w4', generate(kid_in_house_merge)
tab kid_in_house_merge



drop if kid_in_house_merge == 2


sort kid_id
merge m:1 kid_id using `kids_out_house_w4', generate(kid_out_house_merge) update 
tab kid_out_house_merge


drop if kid_out_house_merge == 2


//For alive and in house
gen within_18 = 1 if live_in_house_18_ind == 1 & ich03_cd < 51
replace within_18 = 0 if live_in_house_over18_ind == 1 & ich03_cd < 51

gen between_12to24 = 1 if live_in_house_12to24_ind == 1 & ich03_cd < 51
gen within_24 = 1 if live_in_house_24_ind == 1 & ich03_cd < 51
gen within_21 = 1 if live_in_house_21_ind == 1 & ich03_cd < 51
gen between_12to21 = 1 if live_in_house_12to21_ind == 1 & ich03_cd < 51

//For alive and out of house
replace within_18 = 1 if live_out_house_18_ind == 1 & ich03_cd == 51
replace within_18 = 0 if live_out_house_over18_ind == 1 & ich03_cd == 51

replace between_12to24 = 1 if live_out_house_12to24_ind == 1 & ich03_cd == 51
replace within_24 = 1 if live_out_house_24_ind == 1 & ich03_cd == 51
replace within_21 = 1 if live_out_house_21_ind == 1 & ich03_cd == 51
replace between_12to21 = 1 if live_out_house_12to21_ind == 1 & ich03_cd == 51

rename id  mother_id
gen id = kid_id

**If mother_id_ind == 0 then "id" is kid's id, if mother_id_ind == 1 then "id" is mother's id 
gen mother_id_ind = 0
gen dead = 0
gen dead_0to28_days = 0
gen dead_0to12_mons = 0
gen dead_1to12_mons = 0
gen within_24_old_version = 1



**We have 21 observations which are neither out of household nor dead, but not match with live in household
drop if within_24 == .

tempfile kids_alive_w4
save `kids_alive_w4', replace 



*********************************************************
//Get info on children born live but then died from I_RH
*********************************************************

use "Wave IV/I_RH13", clear
sort iid

**The survey dates are between Oct-Dec 2009, so any children born before Oct. 2007 will be beyond 24 months
**We drop these observations for each of analysis below

destring irh13_yy irh13_mm, replace 
keep if irh13_yy >= 2012 | (irh13_yy == 2011 & irh13_mm >=9  & irh13_mm < .)
drop if irh13_yy >= .

sort iid
//Merge info on survey date
merge m:1 iid using `survey_date_info_w4'
tab _merge
keep if _merge == 3



gen elapsed_mon = survey_mon - birth_mon
gen death_elapsed_diff = age_died_mon - elapsed_mon
tab death_elapsed, m



//Generate dummies for whether mother/child id and if dead/alive
gen mother_id_ind = 1
gen dead = 1

gen dead_0to28_days = 1 if age_died_days <= 28
replace dead_0to28_days = 0 if age_died_days > 28 & age_died_days < .

gen dead_0to12_mons = 1 if age_died_mon <= 12
replace dead_0to12_mons = 0 if age_died_mon > 12 & age_died_mon < .

gen dead_1to12_mons = 1 if age_died_mon >= 1 & age_died_mon <= 12
replace dead_1to12_mons = 0 if (age_died_mon < 1 |(age_died_mon > 12 & age_died_mon < .))


gen within_18 = 1 if elapsed_mon <= 18
replace within_18 = 0 if elapsed_mon > 18 & elapsed_mon < .
gen within_24 = 1 if elapsed_mon <= 24
replace within_24 = 0 if elapsed_mon > 24 & elapsed_mon < .
gen within_21 = 1 if elapsed_mon <= 21
replace within_21 = 0 if elapsed_mon > 21 & elapsed_mon < .

gen between_12to24 = 1 if elapsed_mon > 12 & elapsed_mon <= 24
replace between_12to24 = 0 if elapsed_mon <=12 | (elapsed_mon > 24 & elapsed_mon < .)
gen between_12to21 = 1 if elapsed_mon > 12 & elapsed_mon <= 21
replace between_12to21 = 0 if elapsed_mon <= 12 | (elapsed_mon > 21 & elapsed_mon < .)

gen born_2011 = (irh13_yy == 2011)


keep iid survey_mon birth_mon birth_year dead* mother_id age_died_mon age_died_days elapsed_mon survey_year within* between_12to24 between_12to21 born_2011
rename iid id


****


append using `kids_alive_w4'


gen dead_0to28_days_last18 = dead_0to28_days if within_18 == 1
gen dead_0to28_days_last24 = dead_0to28_days if within_24 == 1

gen dead_0to12_mons_last24 = dead_0to12_mons if within_24 == 1
gen dead_1to12_mons_last24 = dead_1to12_mons if within_24 == 1
gen dead_0to12_mons_12to24 = dead_0to12_mons if between_12to24 == 1

gen dead_0to12_mons_last21 = dead_0to12_mons if within_21 == 1
gen dead_0to12_mons_12to21 = dead_0to12_mons if between_12to21 == 1


//Generate variable identifying household (absent of 4th digit specific to book) for future merges
gen temp_code = substr(id,12,2)
gen temp_indicator = 1 if temp_code == "13" 

gen rid_w4 = substr(id,1,3) + "1" + substr(id,5,13)
replace rid_w4 = substr(id, 1, 3) + "1" + substr(id,5,7) + "00" + substr(id,14,4) if temp_indicator == 1
sort rid_w4
gen rid = substr(rid_w4,1,7) + substr(rid_w4,10,2)

gen survey_round = 2
drop temp_code temp_indicator

tempfile infant_mortality_w4
save `infant_mortality_w4', replace



*** 4. APPEND ALL WAVES ***
//Use Wave I merged file 
use `infant_mortality_w1', clear

//Append Wave III and IV observations 
append using `infant_mortality_w3' `infant_mortality_w4', force
sort rid survey_round 

tempfile infantmortality_allwaves 
save `infantmortality_allwaves', replace


*** 5. CODING PKH RANDOMIZATION/TREATMENT STATUS ***
use "Wave I/Data/pkhben2013_use.dta", clear

//kep only ea, randomization, PKH status, province code
keep ea L07 K09 K13 prov09 

//keep one observation per ea (kecamatan)
duplicates drop ea, force

//drop extra 90 kecs in order to restrict to original 360 kecamatans
drop if missing(L07)

tempfile ea_treatment_list
save `ea_treatment_list'

//use merged all-wave household data
use `infantmortality_allwaves', clear

//generate ea string
gen ea = substr(rid, 1, 3)

//merge with kecamatan treatment list 
merge m:1 ea using `ea_treatment_list', generate(merge_treat)
tab merge_treat
//only keep observations from original 360 kecamatans (L07 is non-missing)
keep if merge_treat == 3
drop merge_treat
sort rid survey_round


*** 6. MERGING HOUSEHOLD-LEVEL COVARIATES ***
tempfile mortality_allwaves_coded
save `mortality_allwaves_coded', replace 

//load master household dataset 
cd "`PKH'"
use "data/coded/household_allwaves_master.dta", clear 

//keep only relevant household covariates 
keep rid* survey_round received* pkh* L07 hh_head_education hh_head_agr ///
	 hh_head_serv hh_educ* roof_type* wall_type* floor_type* clean_water own_latrine ///
	 square_latrine own_septic_tank electricity_PLN logpcexp* hhsize_ln *baseline_nm *miss ///
	 kabupaten kecamatan province split_indicator

tempfile hh_covariates
save `hh_covariates', replace

//merge 
use `mortality_allwaves_coded', clear 
gen rid_merge = rid if survey_round == 0
replace rid_merge = rid_w3 if survey_round == 1
replace rid_merge = rid_w4 if survey_round == 2
merge m:1 rid_merge survey_round using `hh_covariates', generate(_merge_hh)
tab _merge_hh
drop if _merge_hh == 2
drop _merge_hh


*** 7. GENERATE KECAMATAN-LEVEL BASELINE AVERAGES OF OUTCOMES ***
//for right now, only outcomes that appear in regression tables 
local outcomes dead_0to28_days_last24 dead_1to12_mons_last24

//generate kecamatan-level averages of outcomes
foreach x of varlist `outcomes' {
	bysort kecamatan survey_round: egen `x'_kecw = mean(`x')
	bysort kecamatan (survey_round): gen `x'_kecbl = `x'_kecw[1] if survey_round[1] == 0
}


//save
compress
cd "`PKH'data/coded"
save "infantmortality_allwaves_master", replace


