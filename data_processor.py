"""
This data processor file is for the data preprocessing,
it includes data cleaning, data transformation and translation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor():
    """
    A class for loading and preprocessing the data.
    """
    def __init__(self, origin_file, target_file, degrees):
        self.degrees = degrees
        self.data = self.load_data(origin_file)
        self.translate_headers()
        self.preprocessing()
        self.save_data(target_file)
        
    def load_data(self, file_path):
        """
        Load data from the file and return it as a pandas dataframe.
        """
        print(f'Load data from {file_path}...')
        data = pd.read_csv(
            file_path,
            on_bad_lines = 'skip',
            delimiter = ';',
            skiprows = 0,
            low_memory = False,
            dtype = {
                'ano_curriculo': str,
                'cod_curriculo': str,
                'mat_ano': 'Int8',
                'mat_sem': 'Int8',
                'periodo': str,
                'ano': str,
                'semestre': 'Int8',
                'semestre_recomendado': 'Int8',
                'semestre_do_aluno': 'Int8',
                'no_creditos': 'Int8',
                'cep': str,
                'matricula': str,
                'puntos_enem': np.float32,
                'diff': 'Int8',
                'tentativas': 'Int8',
                'cant': 'Int8',
                'cod_enfase': str
            },
            converters = {'grau': lambda x: x.replace(',', '.')},  # change ',' in grades to '.'
        )
        
        return data
    
    def translate_headers(self):
        """
        Translate the Portuguese headers of the data to English.
        """
        self.data = self.data.rename(columns = {
            'cod_curso': 'college code',
            'nome_curso': 'college name',
            'cod_hab': 'degree code',
            'nome_hab': 'degree name',
            'cod_enfase': 'emphasis code',
            'nome_enfase': 'emphasis name',
            'ano_curriculo': 'curriculum year',
            'cod_curriculo': 'curriculum code',
            'matricula': 'student identifier',
            'mat_ano': 'enrolment year',
            'mat_sem': 'enrolment semester',
            'periodo': 'term',
            'ano': 'year',
            'semestre': 'semester',
            'grupos': 'groups',
            'disciplina': 'course',
            'semestre_recomendado': 'recommended semester',
            'semestre_do_aluno': 'student semester',
            'no_creditos': 'credits',
            'turma': 'class',
            'grau': 'grades',
            'sit_final': 'final status',
            'sit_vinculo_atual': 'current status',
            'nome_professor': 'professor name',
            'cep': 'cep (o)',
            'puntos_enem': 'puntos_enem (o)',
            'diff': 'diff',
            'tentativas': 'attempts',
            'cant': 'cant (o)',
            'count': 'count (o)',
            'identificador': 'identifier',
            'nome_disciplina': 'discipline name'})     

    def preprocessing(self):
        """
        This function preprocesses the data by filtering, cleaning, normalizing, and validating it.
        Returns: The preprocessed data
        """
        print('Start preprocessing data...')
        # Filter by degrees ['ARQ', 'ADM', 'CSI'] or one specific degree
        self.data = self.data[self.data['college code'].isin(self.degrees)]

        # Remove leading/trailing white space from string columns
        self.data = self.data.applymap(lambda x: x.strip() if isinstance(x, str) else x) 

        # Sort by student identifier and then term
        self.data = self.data.sort_values(['student identifier', 'term'])

        # Drop irrelevant columns 
        drop_cols = ['college name', 'degree code', 'degree name', 'emphasis code', 'emphasis name',
                     'curriculum year', 'year', 'semester', 'cep (o)', 'cant (o)', 'count (o)', 
                     'puntos_enem (o)', 'groups', 'identifier', 'discipline name']
        self.data = self.data.drop(drop_cols, axis=1)

        # Convert categorical data to numerical data
        le_cols = ['college code', 'student identifier', 'final status', 'course', 'class', 'professor name']
        for i in le_cols:
            self.data[i] = LabelEncoder().fit_transform(self.data[i])

        # Categorize current status to 0/1
        self.data['current status'] = self.data['current status'].replace(['DESLIGADO','MATRICULA EM ABANDONO','JUBILADO'], 0) # fail
        self.data['current status'] = self.data['current status'].replace(['FORMADO','MATRICULADO','MATRICULADO','HABILITADO A FORMATURA',
                                                                            'MATRICULA DESATIVADA','MATRICULA TRANCADA','TRANSFERIDO PARA OUTRA IES',
                                                                            'EM ADMISSAO','MATRICULADO EM CONVENIO','FALECIDO'], 1) # pass

        
        self.data = self.data.apply(pd.to_numeric)
        
        # Replace missing grades as zeros 
        self.data['grades'] = self.data['grades'].fillna(0)

        # Remove missing values
        self.data.dropna(inplace = True)

        # Scale numerical data
        scaler = StandardScaler()
        for col in self.data.columns:
            self.data[col] = scaler.fit_transform(self.data[col].values.reshape(-1, 1))

    def save_data(self, file_path):
        """
        Save the preprocessed data to the target_file.
        """
        self.data.to_csv(file_path, index = False) 
        print(f'Preprocessed data is saved in {file_path}.')
        print('Data preprocessing finished!')

# DataPreprocessor('data/historicosFinal.csv', 'data/ADM_preprocessed.csv', ['ADM'])
# DataPreprocessor('data/historicosFinal.csv', 'data/CSI_preprocessed.csv', ['CSI'])
# DataPreprocessor('data/historicosFinal.csv', 'data/ARQ_preprocessed.csv', ['ARQ'])
DataPreprocessor('data/historicosFinal.csv', 'data/all_preprocessed.csv', ['ARQ', 'ADM', 'CSI'])

