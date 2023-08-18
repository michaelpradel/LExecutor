import os
import pandas as pd

folder_path = "./metrics/"

# Get a list of used datasets
datasets = os.listdir(folder_path)

for dataset in datasets:
    # Get a list of used predictors
    predictors = os.listdir(folder_path + dataset)

    combined_df_for_dataset = pd.DataFrame()
    
    for predictor in predictors:
        # Get a list of raw metric files
        files = os.listdir(f'{folder_path}{dataset}/{predictor}/raw') 
        
        all_executions_df = pd.DataFrame()

        for execution in range(1, 11):

            # Filter the files to include only the ones that match the pattern "metrics_x.csv"
            matching_files = [file for file in files if file.startswith("metrics_") and file.endswith(f"_{execution}.csv")]

            combined_df_for_predictor = pd.DataFrame()

            for file in matching_files:
                try:
                    df = pd.read_csv(f'{folder_path}{dataset}/{predictor}/raw/{file}')
                    combined_df_for_predictor = pd.concat([combined_df_for_predictor, df], ignore_index=True)
                except pd.errors.EmptyDataError:
                    print(file)

            files = combined_df_for_predictor.file.unique()

            for file in files:
                indexes = combined_df_for_predictor.index[combined_df_for_predictor['file'] == file].tolist()
                for index in indexes[:-1]:
                    combined_df_for_predictor = combined_df_for_predictor.drop(index=index)

            all_executions_df = pd.concat([all_executions_df, combined_df_for_predictor], ignore_index=True)

        combined_df_for_predictor = all_executions_df.groupby('file', as_index=False)["covered_iids","total_uses","guided_uses","covered_lines","executed_lines", "execution_time", "random_predictions","type4py_predictions"].mean()
        combined_df_for_predictor['predictor'] = [predictor] * len(combined_df_for_predictor)

        if predictor == 'PynguinTests':
            aux_df = pd.read_csv("wrapp_info.csv")

            combined_df_for_predictor = combined_df_for_predictor.merge(aux_df, on='file', how='left')
            combined_df_for_predictor['covered_lines'] = combined_df_for_predictor['covered_lines'] - combined_df_for_predictor['wrapped'] - 1
            combined_df_for_predictor['covered_lines'] = combined_df_for_predictor.apply(lambda x: x['covered_lines'] if x['covered_lines']>=0 else 0, axis=1)

            combined_df_for_predictor['executed_lines'] = combined_df_for_predictor['executed_lines'] - combined_df_for_predictor['wrapped'] - 1
            combined_df_for_predictor['executed_lines'] = combined_df_for_predictor.apply(lambda x: x['executed_lines'] if x['executed_lines']>=0 else 0, axis=1)
            combined_df_for_predictor.drop(['wrapped'], inplace=True, axis=1)

            combined_df_for_predictor['file'] = combined_df_for_predictor['file'].str.replace("pynguin_tests/test_", "", regex=True)
            combined_df_for_predictor['file'] = combined_df_for_predictor['file'].str.replace("[_]", "/", regex=True)
            combined_df_for_predictor['file'] = combined_df_for_predictor['file'].str.replace("popular/projects/snippets/dataset", "popular_projects_snippets_dataset", regex=True)
            combined_df_for_predictor['file'] = combined_df_for_predictor['file'].str.replace("functions", "bodies", regex=True)
            combined_df_for_predictor['file'] = combined_df_for_predictor['file'].str.replace("function/", "body_", regex=True)

        elif predictor == 'Type4PyValuePredictor':
            aux_df = pd.read_csv("aux_data_functions_with_invocation_dataset.csv")

            combined_df_for_predictor = combined_df_for_predictor.merge(aux_df, on='file', how='left')
            combined_df_for_predictor['covered_lines'] = combined_df_for_predictor['covered_lines'] - combined_df_for_predictor['lines_to_discard']
            combined_df_for_predictor['covered_lines'] = combined_df_for_predictor.apply(lambda x: x['covered_lines'] if x['covered_lines']>=0 else 0, axis=1)

            combined_df_for_predictor['executed_lines'] = combined_df_for_predictor['executed_lines'] - combined_df_for_predictor['lines_to_discard']
            combined_df_for_predictor['executed_lines'] = combined_df_for_predictor.apply(lambda x: x['executed_lines'] if x['executed_lines']>=0 else 0, axis=1)
            combined_df_for_predictor.drop(['lines_to_discard'], inplace=True, axis=1)

        combined_df_for_predictor.to_csv(f'{folder_path}{dataset}/{predictor}/metrics.csv', index=False)

        combined_df_for_dataset = pd.concat([combined_df_for_dataset, combined_df_for_predictor], ignore_index=True)
        
    combined_df_for_dataset['file'] = combined_df_for_dataset['file'].str.replace("functions", "bodies", regex=True)
    combined_df_for_dataset['file'] = combined_df_for_dataset['file'].str.replace("function", "body", regex=True)
    combined_df_for_dataset.to_csv(f'{folder_path}metrics_{dataset}_dataset.csv', index=False)