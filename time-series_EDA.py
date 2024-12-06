"""
=========================
Time series analysis and EDA
=========================
"""
train =sample_df
train['N']=train.index.values
train['id']=train.index.values

xx= sample_df[(sample_df.symbol_id==1)] ['id']
yy=sample_df[ (sample_df.symbol_id==1)]['responder_6']

plt.figure(figsize=(16, 5))
plt.plot(xx,yy, color = 'black', linewidth =0.05)
plt.suptitle('Returns, responder_6', weight='bold', fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Returns", fontsize=12)
plt.grid(color = gridColor , linewidth=0.8)
plt.axhline(0, color='red', linestyle='-', linewidth=1.2)
plt.show()

#for symbol_id=1
plt.figure(figsize=(14, 4))
plt.plot(xx,yy.cumsum(), color = 'black', linewidth =0.6)
plt.suptitle('Cumulative responder_6', weight='bold', fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Cumulative res", fontsize=12)
plt.yticks(np.arange(-500,1000,250))
#plt.xticks(np.arange(0,170,10))
plt.grid(color = gridColor)
#plt.grid(color = 'lightblue')
plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
plt.show()

# for symbol_id == 0
plt.figure(figsize=(18, 7))
predictor_cols = [col for col in sample_df.columns if 'responder' in col]
for i in predictor_cols:
    if i == 'responder_6':
        c='red'
        lw=2.5
        plt.plot((sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[i].mean()).cumsum(), linewidth = lw, color = c)
    else:
        lw=1
        plt.plot((sample_df[sample_df.symbol_id == 0].groupby(['date_id'])[i].mean()).cumsum(), linewidth = lw)

plt.xlabel('Trade days')
plt.ylabel('Cumulative response')
plt.title('Response time series over trade days  \n Responder 6 (red) and other responders', weight='bold')
plt.grid(visible=True, color = gridColor, linewidth = 0.7)
plt.axhline(0, color='blue', linestyle='-', linewidth=1)
plt.legend(predictor_cols)
sns.despine()
#plt.show()

plt.figure(figsize=(6, 6))
responders = pd.read_csv(f"{path}/responders.csv")
matrix = responders[[ f"tag_{no}" for no in range(0,5,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5,
            linecolor='white', annot=True, fmt='.2f')
plt.xlabel("Responder_0 - Responder_8")
plt.ylabel("Responder_0 - Responder_8")
plt.show()

df_train=sample_df
s_id = 0                        # Change params to take a look at other symbols
res_columns = [col for col in df_train.columns if re.match("responder_", col)]
row = 9
j = 0

fig, axs = plt.subplots(figsize=(18, 4*row))
for i in range(1, 3 * len(res_columns) + 1, 3):
    xx= sample_df[(sample_df.symbol_id==s_id)] ['N']
    yy=sample_df[ (sample_df.symbol_id==s_id)][f'responder_{j}']
    c='black'
    if j == 6: c='red'

    ax1 = plt.subplot(9, 3, i)
    ax1.plot(   xx,yy.cumsum()   , color = c, linewidth =0.8 )
    plt.axhline(0, color='blue', linestyle='-', linewidth=0.9)
    plt.grid(color =gridColor )

    ax2 = plt.subplot(9, 3, i+1)
    #by_date = df_symbolX.groupby(["date_id"])
    ax2.plot(xx,yy   , color = c, linewidth =0.05)
    plt.axhline(0, color='blue', linestyle='-', linewidth=1.2)
    ax2.set_title(f"responder_{j}", fontsize = 14)
    plt.grid(color = gridColor)

    ax3 = plt.subplot(9, 3, i+2)
    b=1000
    ax3.hist(yy, bins=b, color = c,density=True, histtype="step" )
    ax3.hist(yy, bins=b, color = 'lightgrey',density=True)
    plt.grid(color = gridColor)
    ax3.set_ylim([0, 3.5])
    ax3.set_xlim([-2.5, 2.5])

    j = j + 1

fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')
plt.show()

res_columns = [col for col in df_train.columns if re.match("responder_", col)]
row=10
fig, axs = plt.subplots(figsize=(18, 5*row))
b=300
j = 0
for i in range(1, 3 * row + 1, 3):
    xx= sample_df[(sample_df.symbol_id==j)] ['N']
    yy= sample_df[(sample_df.symbol_id==j)]['responder_6']
    c='black'

    ax1 = plt.subplot(row, 3, i)
    ax1.plot(   xx,yy.cumsum()   , color = c, linewidth =0.8 )
    plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
    plt.grid(color = gridColor)
    plt.xlabel('Time')

    ax2 = plt.subplot(row, 3, i+1)
    ax2.plot(xx,yy   , color = c, linewidth =0.05)
    plt.axhline(0, color='red', linestyle='-', linewidth=0.7)
    ax2.set_title(f"symbol_id={j}", fontsize = '14')
    plt.grid(color = gridColor)
    plt.xlabel('Time')

    ax3 = plt.subplot(row, 3, i+2)
    ax3.hist(yy, bins=b, color = c, density=True, histtype="step" )
    ax3.hist(yy, bins=b, color = 'lightgrey',density=True)
    plt.grid(color = gridColor)
    ax3.set_xlim([-2.5, 2.5])
    ax3.set_ylim([0, 1.5])
    plt.xlabel('Time')

    j = j + 1

fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')
plt.show()

df_train = sample_df
plt.figure(figsize=(20, 3))    # Plot missing values
plt.bar(x=df_train.isna().sum().index, height=df_train.isna().sum().values, color="red", label='missing')   # analog: using missingno
plt.xticks(rotation=90)
plt.title(f'Missing values over the {len(df_train)} samples which have a target')
plt.grid()
plt.legend()
plt.show()

features = pd.read_csv(f"{path}/features.csv")
features

plt.figure(figsize=(18, 6))
plt.imshow(features.iloc[:, 1:].T.values, cmap="gray_r")
plt.xlabel("feature_00 - feature_78")
plt.ylabel("tag_0 - tag_16")
plt.yticks(np.arange(17))
plt.xticks(np.arange(79))
plt.grid(color = 'lightgrey')
plt.show()

plt.figure(figsize=(11, 11))
matrix = features[[ f"tag_{no}" for no in range(0,17,1) ] ].T.corr()
sns.heatmap(matrix, square=True, cmap="coolwarm", alpha =0.9, vmin=-1, vmax=1, center= 0, linewidths=0.5, linecolor='white')
plt.show()


responders = pd.read_csv(f"{path}/responders.csv")
responders


sample_df['weight'].describe().round(1)

plt.figure(figsize=(8,3))
plt.hist(sample_df['weight'], bins=30, color='grey', edgecolor = 'white',density=True )
plt.title('Distribution of weights')
plt.grid(color = 'lightgrey', linewidth=0.5)
plt.axvline(1.7, color='red', linestyle='-', linewidth=0.7)
plt.show()


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

gs=600
k=1;
col = 3
row = 3
fig, axs = plt.subplots(row, col, figsize=(5*col, 5*row))

for i in numerical_features:

    plt.subplot(col,row, k)
    plt.hexbin(sample_df[i], sample_df['responder_6'], gridsize=gs, cmap='CMRmap', bins='log', alpha = 0.2)
    plt.xlabel(f'{i}', fontsize = 12)
    plt.ylabel('responder_6', fontsize = 12)
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    k=k+1
fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')

plt.show()

numerical_features=[]
for i in ['05', '06', '07', '08', '12', '15', '19', '32', '38', '39', '50', '51', '65', '66', '67']:
    numerical_features.append(f'feature_{i}')

gs=600
k=1;
col = 3
row = int(np.ceil(len(numerical_features) /3 ))
sz=5
w=sz*col
h = w/col *row
plt.figure(figsize=(w, h))

fig, axs = plt.subplots(figsize=(w, h))

for i in numerical_features:

    plt.subplot(row, col, k)
    plt.hexbin(sample_df['responder_6'], sample_df[i], gridsize=gs, cmap='CMRmap', bins='log', alpha = 0.3)

    plt.xlabel(f'{i}')
    plt.ylabel('responder_6')
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    k=k+1

fig.patch.set_linewidth(3)
fig.patch.set_edgecolor('#000000')
fig.patch.set_facecolor('#eeeeee')
plt.show()

numerical_features=[]

for i in range(5,9):
    numerical_features.append(f'feature_0{i}')
for i in range(15,20):
    numerical_features.append(f'feature_{i}')

a=0; k=1;
n=3;

fig, axs = plt.subplots(figsize=(15, 4))
for i in numerical_features[:-1]:
    a=a+1
    for j in numerical_features[a:]:
        plt.subplot(1,n, k)
        plt.hexbin(sample_df[i], sample_df[j], gridsize=200, cmap='CMRmap', bins='log', alpha = 1)
        plt.grid()
        plt.xlabel(f'{i}', fontsize = 14)
        plt.ylabel(f'{j}', fontsize = 14)
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        k=k+1
        if k == (n+1):
            k=1
            plt.show()
            plt.figure(figsize=(15, 4))