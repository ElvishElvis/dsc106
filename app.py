import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt

st.title('DSC 106 Final Project')
st.text('DSC 106 Final Project')
st.text('Yuye Huang A14796379')
st.text('Jia Shi A15233744')

from PIL import Image
for i in range(1,3):
	image = Image.open('./presentation/presentation (1).00{}.jpeg'.format(i))
	st.image(image, caption=' ',use_column_width=True)



df = pd.read_csv('US_graduate_schools_admission_parameters_dataset.csv')
corr=df.drop('Serial No.', axis=1).corr()

df['Research']=df['Research'].apply(lambda a: 'have research' if a==1 else 'no reserach')


sns.set(style="white")
# Generate a mask for the upper triangle
mask0 = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 11))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
#sns.set(font_scale=2)
sns.heatmap(corr,mask=mask0, cmap=cmap, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True, fmt=".2f",)

ax.text(0,0,'Correlation between graduate school admission and other factors',fontsize=22)
#ax.title('Correlation of different factors with graduate school admission')
st.write('### GPA IS MOST IMPORTANCE?')
f.savefig('admission correlation heatmap.jpg')
st.pyplot()


st.write('### HIGHER GPA, BETTER CHANCE?')
f, ax = plt.subplots(figsize=(8,6))
sns.lineplot(df['CGPA'],df['Chance of Admit '],linewidth=2.5)
plt.xticks(rotation=90)
plt.show()

f.savefig('gpa line graph',bbox_inches='tight')
st.pyplot()


st.write('### WHAT ABOUT RATING OF UNDERGRADUATE SCHOOL?')

f, ax = plt.subplots(figsize=(8,6))
sns.lineplot(df['University Rating'],df['Chance of Admit '],linewidth=2.5)
plt.xticks(rotation=90)
plt.show()

f.savefig('rating line graph',bbox_inches='tight')
st.pyplot()

st.write('### BETTER COLLEGE->HIGHER GPA->GREATER CHANCE OF GET INTO GRAD-SCHOOL？')


f, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(y="Chance of Admit ", x="CGPA",
                hue="University Rating",
                data=df)
f.savefig('gpa vs rating vs admit',bbox_inches='tight')
st.pyplot()

st.write('### WHAT ABOUT RESEARCH? (LOWEST CORRELATION)')
sns.boxplot(x="Research", y="Chance of Admit ", data=df)
plt.savefig('research boxplot',bbox_inches='tight')
st.pyplot()


st.write('### RESEARCH->HIGHER GPA->GREATER CHANCE OF GET INTO GRAD-SCHOOL？')
#f, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x="CGPA", y="Chance of Admit ",
                hue="Research",
                data=df)

plt.savefig('research vs admit vs gpa',bbox_inches='tight')
st.pyplot()
















file=pd.read_csv('US_graduate_schools_admission_parameters_dataset.csv')
file=file.drop(['Serial No.'],axis=1)
# sns.heatmap(file.corr())
df=file.sort_values(['Chance of Admit '],ascending=False)
import pandas as pd
from sklearn import preprocessing
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research','Chance of Admit ']

df['Standard_Score']=df[['GRE Score','TOEFL Score','CGPA']].sum(axis=1)
df['Other Factors']=df[['University Rating','SOP','LOR ','Research']].sum(axis=1)
dk=df[['Chance of Admit ','Standard_Score','Other Factors']]
x = dk.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dk = pd.DataFrame(x_scaled)
dk.columns=['Chance of Admit ','Standard_Score','Other Factors']
# dk['Chance of Admit ']=dk['Chance of Admit '].apply(lambda a: 2 if a>0.75 else( 1 if a>0.5 else 0 ))

st.write('### Enemble Normized Vectors Vs Admission Rate ')

st.write('Standard_Score: GRE, GPA, TOEFL')
st.write('Other Factors: University Rating, Statement of Purpose, Letter of Recommendation, Research')

c = alt.Chart(dk).mark_circle().encode(x='Standard_Score', y='Other Factors', size='Chance of Admit ', color='Chance of Admit ')
st.altair_chart(c, use_container_width=True)

# df = pd.read_csv('US_graduate_schools_admission_parameters_dataset.csv')
# corr=df.drop('Serial No.', axis=1).corr()
# sns.set(style="white")
# # Generate a mask for the upper triangle
# mask0 = np.triu(np.ones_like(corr, dtype=np.bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 11))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# #sns.set(font_scale=2)
# sns.heatmap(corr,mask=mask0, cmap=cmap, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True, fmt=".2f",)

# ax.text(0,0,'Correlation between graduate school admission and other factors',fontsize=22)
# #ax.title('Correlation of different factors with graduate school admission')

# f.savefig('admission correlation heatmap.jpg')
# st.pyplot()



# sns.countplot(x='Research', hue='University Rating', data=df)
# plt.show()


st.write('### CONCLUSTION:')
st.write('1. All of the feature have almost a strong correlation with the admission rate. GPA is the single most \
important feature that correlated to the chance of admission, while the research is relatively least related \
to the admission, one possibility is that people with research experience are applying for better school witch has lower admission possibility')
st.write('2. Student with stronger standard score would also tend to be stronger in other aspect. For example, student with higher GPA have a \
	much higher rate of having research experience. Thus the competition among the top students is very fierce since they would tend to excel \
	in multi aspects. ')
st.write('3. From the result of the ensemble variable correlation, we can see that both of the ensemble variables have a strong correlation \
	with the admission rate, with a slightly stronger correlation to the standard score.')






st.write('### Top Data Science Master Program arosss United States')
df=pd.read_csv('college.csv')
st.map(df)
st.text('Source: https://www.datasciencedegreeprograms.net/rankings/masters-data-science/')
# chart_data=pd.read_csv('norm.csv', index_col=0)

# st.bar_chart(chart_data.to_numpy())


st.write('### Questionaire for Data Science major outcome')
st.write('We would like to invite you to fill out the Questionaire for future student of UCSD undergraduate data science major outcome')

option = st.selectbox('What is your outcome after graduation',('Work', 'Gradudate School', 'Have not decide'))

GPA = st.text_input('What is your GPA of graduation (N/A) if do not want to answer ', '4.0')

GRE = st.text_input('What is your GRE of graduation (N/A) if do not want to answer ', '340')

title = st.text_input('If you are going to continuous education, please enter the name of school and program (N/A) if not apply', 'UCSD/ BSMS')

com = st.text_input('If you are going to work, please enter the name of company and position (N/A) if not apply', 'Entegris/ Data Analyst')




st.write('#### Please check if you agree to have those information for further research')


st.write('Sorry if it is too slow to run the website when click/select anything below')
agree = st.checkbox('I agree')
if agree:
	st.write('Great! Thank you!')



if st.button('Inputed answer'):
	st.write('Your outcome is ', option)
	st.write('Your GPA is ', GPA)
	st.write('Your GRE is ', GRE)

	st.write('Your future work is ', com)

	st.write('Your future school is ', title)


st.write('#### Please check the information above')

if st.button('submit'):
	st.write('Thank you for your participation')



# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)

# for i in range(1, 10):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.05)

# progress_bar.empty()

# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")