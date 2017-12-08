data bank_targetMarketing;
infile '/home/asham0/my_content/fsprojects/DataSets/bank-additional-full.csv' dlm=';' firstobs=2;
input age job$ marital$ education$ default$ housing$ loan$ contact$ month$ day_of_week$ duration campaign pdays previous poutcome$ emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed y$;
run;

proc print data = bank_targetmarketing(obs = 50);
run;

ods graphics on;
proc freq data=bank_targetMarketing order=FORMATTED;
   *tables age / plots=freqplot(type=dotplot);
   tables age*y / plots=freqplot(type=dotplot);
   *tables Educ*NewEsteem / plots=freqplot(type=dotplot scale=percent);
   *weight Count;
   title 'Frequency of outcome per age';
run;
ods graphics off;

data bank_v1;
set bank_targetmarketing; *(drop = contact$ month$ day_of_week$ pdays);
run;

data bank_train;
set bank_v1;
if mod(_n_,2) = 1;
run;

proc means data=bank_train n mean std min max;
*class job marital education default housing loan contact month day_of_week poutcome y;
class y;
*var age duration campaign previous emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed;
title 'Distribution of client subscribed a term deposit (yes/no) in train dataset '
run;

data bank_test;
set bank_v1;
if mod(_n_,2) = 0;
run;

proc means data=bank_test n mean std min max;
class y;
*var age duration campaign previous emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed;
title 'Distribution of client subscribed a term deposit (yes/no) in test dataset '
run;

*age job marital education default housing loan y poutcome emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed;
/* Perform logistic regression first pass on the variables */
proc logistic data=bank_train PLOTS(MAXPOINTS=5000);
*class job marital education default housing loan contact month day_of_week poutcome/ param=ref;
class job marital education default housing loan/ param=ref;
model y= age job marital education default housing loan emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed / selection=FORWARD start=1 scale=none details influence lackfit;
ROC 'MainEffects' age housing emp_var_rate cons_price_idx cons_conf_idx euribor3m ;
*model y(event='yes')= age job marital education default housing loan contact month day_of_week duration campaign pdays previous poutcome emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed/ selection=FORWARD start=1 scale=none details influence lackfit;
*effectplot slicefit(sliceby=Sex plotby=ecg) / noobs;
run;

proc logistic data=bank_test PLOTS(MAXPOINTS=5000);
*class job marital education default housing loan contact month day_of_week poutcome/ param=ref;
class job marital education default housing loan/ param=ref;
model y= age job marital education default housing loan emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed / selection=FORWARD start=1 scale=none details influence lackfit;
ROC 'MainEffects' age housing emp_var_rate cons_price_idx cons_conf_idx euribor3m ;
*model y(event='yes')= age job marital education default housing loan contact month day_of_week duration campaign pdays previous poutcome emp_var_rate cons_price_idx cons_conf_idx euribor3m nr_employed/ selection=FORWARD start=1 scale=none details influence lackfit;
*effectplot slicefit(sliceby=Sex plotby=ecg) / noobs;
run;

/* Repeat the test with only selected varaibles get oddsradio plot and effectplots*/
proc logistic data=bank_train plots(only)=(oddsratio(range=clip));
class job default/ param=ref;
model y= age job default emp_var_rate cons_price_idx cons_conf_idx nr_employed;
oddsratio age;
oddsratio default;
oddsratio emp_var_rate;
oddsratio cons_price_idx;
oddsratio cons_conf_idx;
oddsratio nr_employed;
effectplot / at(Job=all) noobs;
effectplot slicefit(sliceby=Job plotby=default) / noobs;
ROC 'MainEffects' age job default emp_var_rate cons_price_idx cons_conf_idx nr_employed ;
run;
