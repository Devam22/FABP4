import pandas as pd
import numpy as np

names = ['ab','ac']
# for i in ['a','b','c','d']:
#     for j in range(ord('a'), ord('z') + 1):
#         x = i + chr(j)
#         if x == 'db':
#             break;
#         names.append(x)

final_df = pd.DataFrame()
for suffix in names:
    file_name = "Final_FABP4_results"
    file_name = file_name + '_' + suffix + '.csv'
    df = pd.read_csv(file_name)
    df = df[df['probability_class_Activity'] > 0.973694]
    final_df = pd.concat([final_df,df],ignore_index=True)


final_df.to_csv('Final.csv',index=False)

