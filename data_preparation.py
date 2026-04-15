import os
import pandas as pd

def prepare_data(base_path="Data"):
    df_bin_list = []
    df_topic_list = []

    # ---------------------------------------------------------
    # 1. Fake News Detector Spanish News Edition
    # ---------------------------------------------------------
    print("Procesando: Fake News Detector Spanish News Edition...")
    fnd_path = os.path.join(base_path, "Fake News Detector Spanish News Edition", "train.csv")
    if os.path.exists(fnd_path):
        df_fnd = pd.read_csv(fnd_path)
        # Columns: 'title', 'text', 'label'
        # label: 1 = Fake ("False"), 0 = Real ("True")
        
        df_bin = pd.DataFrame()
        df_bin['Title'] = df_fnd['title'].fillna("")
        df_bin['Content'] = df_fnd['text'].fillna("")
        df_bin['State'] = df_fnd['label'].map({1: "False", 0: "True"})
        df_bin['Dataset'] = "Fake News Detector Spanish News Edition"
        df_bin_list.append(df_bin)
    
    # ---------------------------------------------------------
    # 2. Fake-news-latam-omdena
    # ---------------------------------------------------------
    print("Procesando: Fake-news-latam-omdena...")
    latam_dir = os.path.join(base_path, "Fake-news-latam-omdena")
    if os.path.exists(latam_dir):
        for file in ["train-00000-of-00001.parquet", "test-00000-of-00001.parquet"]:
            f_path = os.path.join(latam_dir, file)
            if os.path.exists(f_path):
                df_latam = pd.read_parquet(f_path)
                # Columns: 'Title', 'Text', 'Corrected_label'
                # Corrected_label: 'True' or 'Fake'
                
                df_bin = pd.DataFrame()
                df_bin['Title'] = df_latam.get('Title', pd.Series([""] * len(df_latam))).fillna("")
                df_bin['Content'] = df_latam.get('Text', pd.Series([""] * len(df_latam))).fillna("")
                
                # Mapeo de etiqueta
                state_map = {"True": "True", "Fake": "False"}
                df_bin['State'] = df_latam['Corrected_label'].map(state_map)
                df_bin['Dataset'] = "Fake-news-latam-omdena"
                
                # Descartamos si no hay etiqueta
                df_bin = df_bin.dropna(subset=['State'])
                df_bin_list.append(df_bin)

    # ---------------------------------------------------------
    # 3. Spanish Political Fake News
    # ---------------------------------------------------------
    print("Procesando: Spanish Political Fake News...")
    pol_path = os.path.join(base_path, "Spanish Political Fake News", "D57000_complete.csv")
    if os.path.exists(pol_path):
        df_pol = pd.read_csv(pol_path, sep=';')
        # Columns: 'ID', 'Label', 'Titulo', 'Descripcion', 'Fecha'
        # Label: 1 = Fake ("False"), 0 = True ("True")
        
        df_bin = pd.DataFrame()
        df_bin['Title'] = df_pol['Titulo'].fillna("")
        df_bin['Content'] = df_pol['Descripcion'].fillna("")
        df_bin['State'] = df_pol['Label'].map({1: "False", 0: "True"})
        df_bin['Dataset'] = "Spanish Political Fake News"
        
        df_bin = df_bin.dropna(subset=['State'])
        df_bin_list.append(df_bin)
        
        # Este dataset especificamente es de Politica, así que lo mandamos a Topic Classifier
        df_topic = pd.DataFrame()
        df_topic['Title'] = df_bin['Title']
        df_topic['Content'] = df_bin['Content']
        df_topic['Topic'] = "Politics"
        df_topic['Dataset'] = "Spanish Political Fake News"
        df_topic_list.append(df_topic)

    # ---------------------------------------------------------
    # 4. The Spanish Fake News Corpus
    # ---------------------------------------------------------
    print("Procesando: The Spanish Fake News Corpus...")
    corpus_dir = os.path.join(base_path, "The Spanish Fake News Corpus")
    if os.path.exists(corpus_dir):
        for file in ["train.xlsx", "development.xlsx", "test.xlsx"]:
            f_path = os.path.join(corpus_dir, file)
            if os.path.exists(f_path):
                df_corpus = pd.read_excel(f_path)
                
                # Normalizar los nombres de columnas (test.xlsx tiene mayúsculas)
                df_corpus.columns = [col.upper() for col in df_corpus.columns]
                
                df_bin = pd.DataFrame()
                df_bin['Title'] = df_corpus.get('HEADLINE', pd.Series([""] * len(df_corpus))).fillna("")
                df_bin['Content'] = df_corpus.get('TEXT', pd.Series([""] * len(df_corpus))).fillna("")
                
                # Category: 'Fake' o 'True'
                state_map = {"True": "True", "Fake": "False"}
                df_bin['State'] = df_corpus.get('CATEGORY', pd.Series([None] * len(df_corpus))).map(state_map)
                df_bin['Dataset'] = "The Spanish Fake News Corpus"
                
                df_bin = df_bin.dropna(subset=['State'])
                df_bin_list.append(df_bin)
                
                # Topic 
                if 'TOPIC' in df_corpus.columns or 'TOPICS' in df_corpus.columns:
                    col_topic = 'TOPICS' if 'TOPICS' in df_corpus.columns else 'TOPIC'
                    
                    df_topic = pd.DataFrame()
                    df_topic['Title'] = df_bin['Title']
                    df_topic['Content'] = df_bin['Content']
                    df_topic['Topic'] = df_corpus[col_topic].fillna("Unknown").astype(str).str.capitalize()
                    
                    # Agrupar topics similares (limpieza básica)
                    topic_mapping = {
                        "Política": "Politics",
                        "Politic": "Politics",
                        "Poltics": "Politics",
                        "Sociedad": "Society",
                        "Covid-19": "Health",
                        "Ciencia": "Science",
                        "Deporte": "Sport",
                        "Ambiental": "Environment",
                        "Internacional": "International"
                    }
                    df_topic['Topic'] = df_topic['Topic'].replace(topic_mapping)
                    df_topic['Dataset'] = "The Spanish Fake News Corpus"
                    
                    df_topic_list.append(df_topic)

    # ---------------------------------------------------------
    # UNIÓN FINAL Y CREACIÓN DE IDs
    # ---------------------------------------------------------
    print("Unificando datasets...")
    
    # DataFrame Binario
    if len(df_bin_list) > 0:
        final_bin_df = pd.concat(df_bin_list, ignore_index=True)
        final_bin_df['ID'] = ["BIN_" + str(i) for i in range(len(final_bin_df))]
        # Reordenar columnas a la estructura pedida
        final_bin_df = final_bin_df[["ID", "Title", "Content", "State", "Dataset"]]
        
        # Eliminar duplicados absolutos o filas vacias si se considera
        
        final_bin_df.to_csv("Data_Bin_Classifier.csv", index=False, encoding='utf-8-sig')
        print(f"-> Creado 'Data_Bin_Classifier.csv' con {len(final_bin_df)} registros.")
    else:
        print("No se encontraron datos para el dataset binario.")

    # DataFrame Tópico
    if len(df_topic_list) > 0:
        final_topic_df = pd.concat(df_topic_list, ignore_index=True)
        final_topic_df['ID'] = ["TOP_" + str(i) for i in range(len(final_topic_df))]
        final_topic_df = final_topic_df[["ID", "Title", "Content", "Topic", "Dataset"]]
        
        final_topic_df.to_csv("Data_Topic_Classifier.csv", index=False, encoding='utf-8-sig')
        print(f"-> Creado 'Data_Topic_Classifier.csv' con {len(final_topic_df)} registros.")
        print("Distribución de Topics:")
        print(final_topic_df['Topic'].value_counts())
    else:
        print("No se encontraron datos para el dataset de tópicos.")

if __name__ == "__main__":
    prepare_data()
