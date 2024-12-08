"""
=========================
Preprocessing and EDA
=========================
"""
train =sample_df
train['N']=train.index.values
train['id']=train.index.values

df_train=sample_df

features = pd.read_csv(f"{path}/features.csv")
features

responders = pd.read_csv(f"{path}/responders.csv")
responders


sample_df['weight'].describe().round(1)

sub = pd.read_csv(f"{path}/sample_submission.csv")
print( f"shape = {sub.shape}" )
sub.head(10)


col =[]
for i in range(9):
    col.append(f"responder_{i}")

sample_df[col].describe().round(1)

numerical_features=[]
numerical_features=sample_df.filter(regex='^responder_').columns.tolist() # Separate responders
numerical_features.remove('responder_6')


numerical_features=[]
for i in ['05', '06', '07', '08', '12', '15', '19', '32', '38', '39', '50', '51', '65', '66', '67']:
    numerical_features.append(f'feature_{i}')

numerical_features=[]

for i in range(5,9):
    numerical_features.append(f'feature_0{i}')
for i in range(15,20):
    numerical_features.append(f'feature_{i}')