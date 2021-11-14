import sparknlp
#if you will use jsl, don't run following line
#spark= sparknlp.start() 

import json 
import os

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql import functions as F
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType

#jsl healthcare 
from sparknlp_jsl.annotator import *
import sparknlp_jsl
from sparknlp_display import RelationExtractionVisualizer


import warnings
warnings.filterwarnings('ignore')

params = {"spark.driver.memory":"16G", 
          "spark.kryoserializer.buffer.max":"2000M", 
          "spark.driver.maxResultSize":"2000M"} 

#lisans keyleri streamlit projesine upload etme:
with open('/content/spark_nlp_for_healthcare 4.json') as f:
  license_keys = json.load(f)

spark = sparknlp_jsl.start(license_keys['SECRET'],params=params)


import streamlit as st


#app page configurations(optional):
st.set_page_config(page_title="aemintek app",
                    page_icon=":shark:",
                    layout="centered")


#adding images
col1, col2, col3 = st.columns(3)

with col1:
  st.image("https://repository-images.githubusercontent.com/104670986/2e728700-ace4-11ea-9cfc-f3e060b25ddf")

with col2:
  st.image("data:image/png;base64,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")

with col3:
  st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYXwo_hsFEahbrpY9fy7M1I2fHWKNfjed-YQ&usqp=CAU")


#adding info to sidebar
st.sidebar.markdown("# SparkNLP Sample Implementations")



TEXT= ["Mark Knopfler was born in Glasgow, Scotland, and raised in Blyth, near Newcastle in England, from the age of seven.",
                "Plaese alliow me tao introdduce myhelf, I am a man of waelth und tiaste",
                "We will go to swimming if the ueather is sunny.",
                "A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides.",
                "TORONTO, Canada    A second team of rocketeers competing for the  #36;10 million Ansari X Prize."
                ]


SENTIMENT_TEXT= ["Michael Dorman does great work as a small-town hustler pursued by gangsters.",
                 "A disturbing and suspenseful trip to the dark side.",
                 "A terrible movie as everyone has said. What made me laugh was the cameo appearance by Scott McNealy",
                 "This is a cute little movie. It works as a happy little holiday movie."]


#defining UDF for text pre-processing
@st.cache(allow_output_mutation=True)
def text_correct_pipeline(SELECTED_TEXT):

  documentAssembler= DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

  tokenizer= Tokenizer()\
      .setInputCols(["document"])\
      .setOutputCol("token")

  stemmer= Stemmer()\
      .setInputCols(["token"])\
      .setOutputCol("stem")

  lemmatizer= LemmatizerModel.pretrained("lemma_antbnc","en") \
      .setInputCols(["token"])\
      .setOutputCol("lemma")

  pos= PerceptronModel.pretrained("pos_anc", "en")\
    .setInputCols(["document", "token"])\
    .setOutputCol("pos")

  spellModel= ContextSpellCheckerModel.pretrained("spellcheck_dl")\
    .setInputCols(["token"])\
    .setOutputCol("context_spell_checked")

  nlpPipeline= Pipeline(stages=[ 
                              documentAssembler,
                              tokenizer,
                              stemmer,
                              lemmatizer,
                              pos,
                              spellModel
    ])
  
  empty_df= spark.createDataFrame([[" "]]).toDF("text")
  pipeline_model= nlpPipeline.fit(empty_df)

  lmodel= LightPipeline(pipeline_model)
  lresult= lmodel.fullAnnotate(SELECTED_TEXT)[0]

  token= []
  stem= []
  lemma= []
  pos= []
  spell= []

  for a, b, c, d, e in zip(lresult["token"], lresult["stem"], lresult["lemma"], lresult["pos"], lresult["context_spell_checked"]):
    token.append(a.result)
    stem.append(b.result)
    lemma.append(c.result)
    pos.append(d.result)
    spell.append(e.result)

  df= pd.DataFrame({"token": token, "stem": stem, "lemma": lemma, "pos_tags": pos, "spell": spell})
  
  return df



#sentiment pipeline UDF
@st.cache(allow_output_mutation=True)
def sentiment_pipeline(SELECTED_TEXT):
  documentAssembler= DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

  tokenizer= Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

  spell_checker= ContextSpellCheckerModel.pretrained("spellcheck_dl")\
    .setInputCols(["token"])\
    .setOutputCol("spell_checked")

  word_embeddings= WordEmbeddingsModel.pretrained("glove_100d")\
    .setInputCols(["spell_checked"])\
    .setOutputCol("embeddings")

  sentence_embeddings= SentenceEmbeddings()\
    .setInputCols(["document", "embeddings"])\
    .setOutputCol("sentence_embeddings")\
    .setPoolingStrategy("AVERAGE")

  classifier= SentimentDLModel.pretrained("sentimentdl_glove_imdb")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

  pipeline= Pipeline(stages=[
                           documentAssembler,
                           tokenizer,
                           spell_checker,
                           word_embeddings,
                           sentence_embeddings,
                           classifier
  ])

  empty_df= spark.createDataFrame([[""]]).toDF("text")
  model= pipeline.fit(empty_df)

  lmodel= LightPipeline(model)
  lresult= lmodel.fullAnnotate(SELECTED_SENTIMENT_TEXT)[0]

  sentiment= []
  document= []
  for m, n in zip(lresult["document"], lresult["sentiment"]):
    document.append(m.result)
    sentiment.append(n.result)

  display_df= pd.DataFrame({"text": document, "sentiment_result": sentiment})
  
  return display_df



#ner pipeline UDF
@st.cache(allow_output_mutation=True)
def load_ner_pipeline():
      documentAssembler= DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

      sentenceDetector= SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
          .setInputCols(["document"])\
          .setOutputCol("sentence")

      tokenizer= Tokenizer()\
          .setInputCols(["sentence"])\
          .setOutputCol("token")

      word_embeddings= WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("embeddings")

      clinical_ner= MedicalNerModel.pretrained("ner_clinical_large", "en", "clinical/models")\
        .setInputCols(["sentence", "token","embeddings"])\
        .setOutputCol("ner")

      ner_converter= NerConverter()\
        .setInputCols(["sentence", "token", "ner"])\
        .setOutputCol("ner_chunks")

      pipeline= Pipeline(stages=[
                                documentAssembler,
                                sentenceDetector,
                                tokenizer,
                                word_embeddings,
                                clinical_ner,
                                ner_converter

      ])

      text = '''
                A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ),
                one prior episode of HTG-induced pancreatitis three years prior to presentation ,
                associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .
                Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .
                She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity .
                Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 .
                Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission .
                However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again .
                The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .
             '''


      empty_df= spark.createDataFrame([[""]]).toDF("text")
      model= pipeline.fit(empty_df)
      lmodel= LightPipeline(model)
      lresult= lmodel.fullAnnotate(text)[0]

      chunks= []
      entities= []
      sentence= []
      begin= []
      end= []

      for n in lresult["ner_chunks"]:
        begin.append(n.begin)
        end.append(n.end)
        chunks.append(n.result)
        sentence.append(n.metadata["sentence"])
        entities.append(n.metadata["entity"])

      light_df= pd.DataFrame({"chunks": chunks, "entities": entities,
                        "sentence": sentence, "begin": begin, "end": end})
      


      return light_df


#Relation Extraction pipeline UDF
@st.cache(allow_output_mutation=True)
def load_rl_pipeline():

  documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

  sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

  tokenizer = sparknlp.annotators.Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

  words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

  pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

  ner_tagger = MedicalNerModel()\
    .pretrained("ner_posology", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

  ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

  dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

  reModel = RelationExtractionModel()\
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)

  pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_chunker,
    dependency_parser,
    reModel
  ])


  text = """
    The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also 
    given 1 unit of Metformin daily.
    He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 
    12 units of insulin lispro with meals , and metformin 1000 mg two times a day.
    """
  empty_data = spark.createDataFrame([[""]]).toDF("text")

  model = pipeline.fit(empty_data)
  lmodel= LightPipeline(model)
  results= lmodel.fullAnnotate(text)[0]

#displayn the RL results UDF
  def get_relations_df (results, col='relations'):
    rel_pairs=[]
    for rel in results[col]:
        rel_pairs.append((
           rel.result, 
           rel.metadata['entity1'], 
           rel.metadata['entity1_begin'],
           rel.metadata['entity1_end'],
           rel.metadata['chunk1'], 
           rel.metadata['entity2'],
           rel.metadata['entity2_begin'],
           rel.metadata['entity2_end'],
           rel.metadata['chunk2'], 
           rel.metadata['confidence']
          ))

    rel_df = pd.DataFrame(rel_pairs, columns=['relation','entity1','entity1_begin','entity1_end','chunk1','entity2','entity2_begin','entity2_end','chunk2', 'confidence'])
    return rel_df

  display_result= get_relations_df(results)

  return display_result



#assertion status detection pipeline UDF
@st.cache(allow_output_mutation=True)
def load_assertion_pipeline():
  documentAssembler= DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

  sentenceDetector= SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

  tokenizer= Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

  word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

  ner= MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols(["sentence" ,"token", "embeddings"])\
    .setOutputCol("ner")

  ner_converter= NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

  clinical_assertion= AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"])\
    .setOutputCol("assertions")

  nlpPipeline= Pipeline(stages=[ 
                              documentAssembler,
                              sentenceDetector,
                              tokenizer,
                              word_embeddings,
                              ner,
                              ner_converter,
                              clinical_assertion
  ])

  empty_df= spark.createDataFrame([[""]]).toDF("text")
  model= nlpPipeline.fit(empty_df)

  text = 'Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain'

  light_model= LightPipeline(model)

  light_result= light_model.fullAnnotate(text)[0]

  chunks= []
  entities= []
  status= []

  for m, n in zip(light_result["ner_chunk"], light_result["assertions"]):
    chunks.append(m.result)
    entities.append(m.metadata["entity"])
    status.append(n.result)

    assertion_df= pd.DataFrame({"chunks": chunks, "entities": entities, "assertion_status": status})
    

  return assertion_df


@st.cache(allow_output_mutation=True)
def load_deide_pipeline(di_type="mask"):
  documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  # Sentence Detector annotator, processes various sentences per line

  sentenceDetector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

  # Tokenizer splits words in a relevant format for NLP

  tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

  # Clinical word embeddings trained on PubMED dataset
  word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")


# NER model trained on n2c2 (de-identification and Heart Disease Risk Factors Challenge) datasets)
  clinical_ner= MedicalNerModel.pretrained("ner_deid_generic_augmented", "en", "clinical/models")\
    .setInputCols(["sentence" ,"token" ,"embeddings"])\
    .setOutputCol("ner")

  ner_converter= NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

  mask_deidentification= DeIdentification()\
    .setInputCols(["sentence", "token", "ner_chunk"])\
    .setOutputCol("deidentified")\
    .setMode("Mask")\
    .setReturnEntityMappings(True)

  obfuscation= DeIdentification()\
      .setInputCols(["sentence", "token", "ner_chunk"])\
      .setOutputCol("deidentified")\
      .setMode("obfuscate")\
      .setObfuscateDate(True)\
      .setObfuscateRefSource("faker")

  if di_type=="mask":
    pipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      mask_deidentification
      ])

    empty_data = spark.createDataFrame([[""]]).toDF("text")
    di_model = pipeline.fit(empty_data)

  elif di_type=="obfuscate":
    pipeline = Pipeline(stages=[
      documentAssembler, 
      sentenceDetector,
      tokenizer,
      word_embeddings,
      clinical_ner,
      ner_converter,
      obfuscation
      ])

    empty_data = spark.createDataFrame([[""]]).toDF("text")
    di_model = pipeline.fit(empty_data)

  text ='''
          A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR # 7194334 Date : 01/13/93 . Patient : Oliveira, 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street
        '''

  result = di_model.transform(spark.createDataFrame([[text]]).toDF("text"))

  result_df= result.select(F.explode(F.arrays_zip('sentence.result', 'deidentified.result')).alias("cols")) \
                   .select(F.expr("cols['0']").alias("sentence"), 
                           F.expr("cols['1']").alias("deidentified")).toPandas()


  return result_df  






#asking options
ask_user= st.sidebar.selectbox("Which task do you want?", ("Text Processing", "Sentiment Analysis", "Clinical Named Entity Recognation",
                                                            "Relation Extraction", "Clinical Assertion Status Detection",
                                                            "Clinical Deidentification (Mask/Obfuscate)"))
#info
st.sidebar.markdown("SparkNLP Version: {}".format(sparknlp.version()))
st.sidebar.markdown("Pyspark Version: {}".format(spark.version))


if ask_user=="Clinical Deidentification (Mask/Obfuscate)":

  st.write('''
             A . Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR # 7194334 Date : 01/13/93 . Patient : Oliveira, 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street
           ''')
  
  mode= st.selectbox("Choose the deidentification method", ("Mask", "Obfuscate"))

  if mode=="Mask":
    
    id_result= load_deide_pipeline(di_type="mask")
    st.dataframe(id_result)

  elif mode=="Obfuscate":

    id_result= load_deide_pipeline(di_type="obfuscate")
    st.dataframe(id_result)

  
  ask_detail= st.checkbox("Click the button to learn some details about the models used in this pipeline.")
  if ask_detail:
    st.write(''' 
        Pretrained **'embeddings_clinical'** has been used as a word embedding. \n
        As NER model,  pretrained **'ner_deid_generic_augmented'** model has been used \n
        Deidentification has 2 options: \n
        - Mask: In the mask mode DeIdentificationModel will replace sensetive entities with ner labels. 
        - Obfuscate: In the obfuscation mode DeIdentificationModel will replace sensetive entities with random values of the same type. \n
        You can reach out this models in sparknlp model hub.
    ''')    



if ask_user== "Clinical Assertion Status Detection":

  st.write('Patient has a headache for the last 2 weeks, needs to get a head CT, and appears anxious when she walks fast. No alopecia noted. She denies pain')
  ad_result= load_assertion_pipeline()
  st.dataframe(ad_result)

  ask_detail= st.checkbox("Click the button to learn some details about the models used in this pipeline.")
  if ask_detail:
    st.markdown(''' 
        Pretrained **'embeddings_clinical'** has been used as a word embedding. \n
        As NER model,  pretrained **'ner_clinical'** model has been used. \n
        As AssertionDLModel, pretrained **'assertion_dl'** model has been used. \n
        The classes that assertion model returns are following: \n
        - present
        - absent
        - possible
        - conditional
        - hypothetical
        - associated_with_someone_else \n
        You can reach out this models in sparknlp model hub.
     ''')    



if ask_user=="Relation Extraction":

  st.write(text = """
    The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also 
    given 1 unit of Metformin daily.
    He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 
    12 units of insulin lispro with meals , and metformin 1000 mg two times a day.
    """)
  rl_result= load_rl_pipeline()
  st.dataframe(rl_result)

  ask_detail= st.checkbox("Click the button to learn some details about the models used in this pipeline.")
  if ask_detail:
    st.write(''' 
        Pretrained **'embeddings_clinical'** has been used as a word embedding. \n
        As NER model,  pretrained **'ner_posology'** model has been used since we used **'posology_re'** as an RL model \n
        RL models need part of speech, dependency parser. **'pos_clinical'** has been used as PerceptronModel. \n
        Pretrained 'dependency_conllu' which is pretrained conll dataset has been used as dependency parser. \n
        You can reach out this models in sparknlp model hub.
    ''')    



if ask_user=="Clinical Named Entity Recognation":

  st.write( '''
                A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ),
                one prior episode of HTG-induced pancreatitis three years prior to presentation ,
                associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .
                Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .
                She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity .
                Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 .
                Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission .
                However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again .
                The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .
             ''')
  ner_result= load_ner_pipeline()
  st.dataframe(ner_result)

  ask_detail= st.checkbox("Click the button to learn some details about the models used in this pipeline.")
  if ask_detail:
    st.write(''' 
        Pretrained **'embeddings_clinical'** has been used as a word embedding. \n
        As a NER model,  pretrained **'clinical_ner_large'** model has been used. \n
        You can reach out this models in sparknlp model hub.
    ''')    



elif ask_user=="Text Processing":

  SELECTED_TEXT= st.selectbox("Choose a sample text for process ", TEXT)
  display_result= text_correct_pipeline(SELECTED_TEXT)
  st.dataframe(display_result)

  ask_detail= st.checkbox("Click the button to learn some details about the models used in this pipeline.")
  if ask_detail:
    st.write(''' 
      In this pipeline, pretrained PerceptronModel has been used for detecting part of speech, model can be found in sparknlp model hub as **'pos_anc'**. \n
      Also, pretrained ContextSpellchecker model has been used for correcting the misspellings, this model works as context aware as well as deep learning based. \n
      You can reach out ContextSpellChecker by **'spellcheck_dl'** in sparknlp model hub. 
    ''')



elif ask_user=="Sentiment Analysis":

  SELECTED_SENTIMENT_TEXT= st.selectbox("Choose a sample text for sentiment result ", SENTIMENT_TEXT)
  display_result= sentiment_pipeline(SELECTED_SENTIMENT_TEXT)
  st.dataframe(display_result)

  ask_detail= st.checkbox("Click the button to learn some details about the models used in this pipeline.")
  if ask_detail:
    st.write(''' 
      In this pipeline, pretrained ContextSpellchecker model has been used for correcting the misspellings, this model works as context aware as well as deep learning based. \n
      You can reach out ContextSpellChecker by **'spellcheck_dl'** in sparknlp model hub. \n
      As a word embedding, **'glove_100d'** has been used. \n 
      For sentiment analysis, pretrained SentimentDlModel has been used. Sentiment model pretrained on imdb reviews. You can reach out this model as **'sentimentdl_glove_imdb'**
    ''')





