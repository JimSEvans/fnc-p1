import pandas as pd
random.seed(1)

bodies = pd.DataFrame.from_csv("train/train_bodies.csv")
stances = pd.DataFrame.from_csv("train/train_stances.csv", index_col=None)
inner = pd.merge(stances, bodies, how='inner', left_on='Body ID', right_index=True, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False)
#how_many_each_stance_each_article = stances.groupby(['Body ID', 'Stance']).agg('count')
train = [] test = []

dg = inner[inner['Stance']=='disagree']['Body ID'].tolist()
uniqdg = set(dg)
dg_test_set_size = round(len(uniqdg)*0.15)
dg_test_set = set(random.sample(uniqdg, dg_test_set_size))
dg_train_set = uniqdg - dg_test_set

ag = inner[inner['Stance']=='agree']['Body ID'].tolist()
uniqag_raw = set(ag)
uniqag = uniqag_raw - uniqdg
ag_test_set_size = round(len(uniqag)*0.15)
ag_test_set = set(random.sample(uniqag, ag_test_set_size))
ag_train_set = uniqag - ag_test_set

dc = inner[inner['Stance']=='discuss']['Body ID'].tolist()
uniqdc_raw = set(dc)
uniqdc = uniqdc_raw - uniqag - uniqdg
dc_test_set = set(random.sample(uniqdc, dc_test_set_size))
dc_train_set = uniqdc - dc_test_set

len(dg_test_set)/(len(dg_test_set)+len(dg_train_set))
len(ag_test_set)/(len(ag_test_set)+len(ag_train_set))
len(dc_test_set)/(len(dc_test_set)+len(dc_train_set))

#ur = inner[inner['Stance']=='unrelated']['Body ID'].tolist()
#uniqur = set(ur)
#ur_test_set_size = round(len(uniqur)*0.15)
all_body_ids = inner['Body ID'].tolist()
done = set(all_body_ids) == uniqdg | uniqag | uniqdc
print(done)

train = list(dg_train_set | ag_train_set | dc_train_set)
test = list(dg_test_set | ag_test_set | dc_test_set)

lentest = len(test)
lentrain = len(train)
lentest/(lentrain+lentest)

print(dg_test_set_size + ag_test_set_size + dc_test_set_size)
print(lentest)
#ag_test_set = set(random.sample(uniqag, ag_test_set_size))
#ag_train_set = uniqag - ag_test_set


#
#
#agrees = inner[inner['Stance']=='agree']['Body ID'].tolist()
#uniqa = set(agrees)
#atest_set_size = round(len(uniqa)*0.15)
#atest_set = set(random.sample(uniqa, atest_set_size))
#atrain_set = uniqa - atest_set
#
#discusses = inner[inner['Stance']=='discuss']['Body ID'].tolist()
#uniqdiscuss = set(discusses) - uniqdisagree - uniqa
#discusstest_set_size = round(len(uniqdiscuss)*0.15)
#discusstest_set = set(random.sample(uniqdiscuss, discusstest_set_size))
#discusstrain_set = uniqdiscuss - discusstest_set
#
#unrelated = inner[inner['Stance']=='unrelated']['Body ID'].tolist()
#uniqunrelated = [x for x in list(set(unrelated)) if x in (uniqdisagree | uniqa | uniqdiscuss)]
#unrelatedtest_set_size = round(len(uniqunrelated)*0.15)
#unrelatedtest_set = set(random.sample(uniqunrelated, unrelatedtest_set_size))
#unrelatedtrain_set = uniqunrelated - unrelatedtest_set
