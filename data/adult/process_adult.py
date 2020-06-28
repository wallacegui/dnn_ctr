
import argparse
import sys
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

def process_adult(input_train_data, output_train_data, input_valid_data,
                 output_valid_data):
    df_train = pd.read_csv(
        tf.gfile.Open(input_train_data),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(input_valid_data),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")
    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    #make label
    df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    #concat
    train_shape= df_train.shape[0]
    test_shape= df_test.shape[0]
    df = pd.concat([df_train,df_test],axis=0)
    #CATEGORICAL_COLUMNS
    le = LabelEncoder()
    for column in CATEGORICAL_COLUMNS:
        df[column] = le.fit_transform(list(df[column]))
    df = df[[LABEL_COLUMN]+CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS]
    df_train = df[:train_shape]
    df_test = df[-test_shape:]
    df_train.to_csv(output_train_data,index=False)
    df_test.to_csv(output_valid_data,index=False)
    

def main(_):
  process_adult(FLAGS.input_train_data, FLAGS.output_train_data, FLAGS.input_valid_data,
                 FLAGS.output_valid_data)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  
  parser.add_argument(
      "--input_train_data",
      type=str,
      default="./train.data",
      help="input Path to the training data."
  )
  parser.add_argument(
      "--output_train_data",
      type=str,
      default="./train.proced.data",
      help="output Path to the training data."
  )
  parser.add_argument(
      "--input_valid_data",
      type=str,
      default="./valid.data",
      help="input Path to the valid data."
  )
  parser.add_argument(
      "--output_valid_data",
      type=str,
      default='./valid.proced.data',
      help="output Path to the valid data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)