############################################################################################################
#   create function for age brackets based on percentile rank, take the min/max and display in text
############################################################################################################

def var_bracket(df, pct_rank_col, target_col, pct_rank):

    if pct_rank <= 20:
        min_in_group = df.loc[df[pct_rank_col] <= 20, target_col].min()
        max_in_group = df.loc[df[pct_rank_col] <= 20, target_col].max()

    if (pct_rank > 20) and (pct_rank <= 40):
        min_in_group = df.loc[(df[pct_rank_col] > 20) & (df[pct_rank_col] <= 40), target_col].min()
        max_in_group = df.loc[(df[pct_rank_col] > 20) & (df[pct_rank_col] <= 40), target_col].max()

    if (pct_rank > 40) and (pct_rank <= 60):
        min_in_group = df.loc[(df[pct_rank_col] > 20) & (df[pct_rank_col] <= 40), target_col].min()
        max_in_group = df.loc[(df[pct_rank_col] > 20) & (df[pct_rank_col] <= 40), target_col].max()

    if (pct_rank > 60) and (pct_rank <= 80):
        min_in_group = df.loc[(df[pct_rank_col] > 60) & (df[pct_rank_col] <= 80), target_col].min()
        max_in_group = df.loc[(df[pct_rank_col] > 60) & (df[pct_rank_col] <= 80), target_col].max()

    elif pct_rank > 80:
        min_in_group = df.loc[df[pct_rank_col] > 80, target_col].min()
        max_in_group = df.loc[df[pct_rank_col] > 80, target_col].max()

    text = target_col + " from " + str(min_in_group) + " to " + str(max_in_group)

    return text




# def age_bracket(df, pct_rank):
#     group = "Group 1"
#     if (pct_rank > 20) and (pct_rank <= 40):
#         group = "Group 2"
#     if (pct_rank > 40) and (pct_rank <= 60):
#         group = "Group 3"
#     if (pct_rank > 60) and (pct_rank <= 80):
#         group = "Group 4"
#     elif pct_rank > 80:
#         group = "Group 5"
#
#     return group



# fig2a = make_subplots(cols = 3, rows = 1, specs = [[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}]])
# fig2a = make_subplots(cols = 3, rows = 1)
#
# trace0 = px.box(insurance_df, x = "age_bracket", y = "expenses", points = "all", color = "sex").data
# trace1 = px.box(insurance_df, x = "age_bracket", y = "expenses", points = "all", color = "smoker").data
# trace2 = px.box(insurance_df, x = "age_bracket", y = "expenses", points = "all", color = "region").data
#
# fig2a.add_trace(trace0, row = 1, col = 1)
# fig2a.add_trace(trace1, row = 1, col = 2)
# fig2a.add_trace(trace2, row = 1, col = 3)
#
# fig2a.show()
#
#
# fig2a = make_subplots(rows=1,cols=1,specs=[[{'type':'box'}]])
# fig2a.add_trace(px.box(insurance_df, x = "age_bracket", y = "expenses", points = "all", color = "smoker"))
# fig2a.show()  # most of the boxplots show that the data are positively skewed; lower whisker is shorter than top whisker
#
#
#     # fig2a.show()  # most of the boxplots show that the data are positively skewed; lower whisker is shorter than top whisker
#
# fig2a = px.box(insurance_df, x = "age_bracket", y = "expenses", points = "all", color = "sex", facet_row = "region")
# fig2a.show()  # most of the boxplots show that the data are positively skewed; lower whisker is shorter than top whisker
#
# fig2b = px.box(insurance_df, x = "bmi_bracket", y = "expenses", points = "all", color = "region")
# fig2b.show()  # positively skewed boxplots observed for people with high BMI from 25 onwards; especially the males
#
#
# fig2b = px.box(insurance_df, x = "expenses")
# # fig2b.show()
#
# #- sub-plots for bmi (numerical variables) against expenses, but coloured with each of the 3 categorical variables
# #- sub-plots for children (numerical variables) against expenses, but coloured with each of the 3 categorical variables

#- Quantile-Quantile (QQ) plot for checking the distribution of a data sample
'''
- NOTE: it is not necessary to transform observed variables (independent and dependent variables) if they do not follow 
- a normal distribution; one should instead check for normality of errors after modelling 
- In linear regression, errors are assumed to follow a normal distribution with a mean of zero
- linear regression still works even if the normality assumption is violated; however, when the errors are not normal
- we can no longer use the statistical techniques like hypothesis testing (or analysis of p-values) after linear 
- regression; these techniques 

#------
- a normal QQ plot is a scatterplot of sample quantiles on the y-axis (the vector of N residuals in rank order)
- the purpose of QQ plots is to find out if two sets of data come from the same theoretical distribution
- if the data is normally distributed, the points in the QQ-normal plot will lie on a straight diagonal line
- however, QQ plot is by no means a check to see if the data are normal. Instead, a normal QQ plot is used to check for
- deviation from normality
'''
# fig4a = probplot(insurance_df.age_scaled, dist = "norm", plot = plt)
# fig4b = probplot(insurance_df.bmi_scaled, dist = "norm", plot = plt)
# "age" shows deviation from normality (i.e. the straight line) the histogram for "age" confirms that
# the distribution of the data is not bell-shaped; "bmi" on the other hand conforms with normality


# #- split data into train and test parts
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
# #- create the polynomial regression model
# polyreg = PolynomialFeatures(degree = 3)
# X_poly = polyreg.fit_transform(X_train)
# poly_linreg = LinearRegression().fit(X_poly, y_train)
#
# #- return the fitted values
# y_poly_pred = poly_linreg.predict(X_poly)
# r2_score(y_train, y_poly_pred)

# score_R2_poly = poly_linreg.score(y_poly_pred, y_test)
# score_adjR2_poly = 1 - ((1 - score_R2_poly)*(len(y_test)-1)) / (len(y_test) - X_test.shape[1] - 1)
# print("R-squared score for the poly regression model:", score_R2_poly) # 0.857
# print("Adjusted R-squared score for the poly regression model:", score_adjR2_poly) # 0.848
#
# #- plot the residuals plot using the fitted values of the polynomial regression model
# fig8 = plt.figure()
# fig8.axes[0] = sns.residplot(poly_linreg_fitted_y, df.columns[-1], data = df, lowess = True,
#                              scatter_kws = {'alpha' : 0.5}, line_kws = {'color' : 'red'})
#
