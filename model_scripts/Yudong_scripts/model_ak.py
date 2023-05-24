import autokeras as ak  # type: ignore
import mytools
import pandas as pd
from constants import TARGET_LABEL

_airport = "KSEA"

input_node = ak.StructuredDataInput()
output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
output_node = ak.RegressionHead()(output_node)

# load train and test data frame
train_df, val_df = mytools.get_train_and_test_ds(_airport)

X_train: pd.DataFrame = train_df.drop(columns=[TARGET_LABEL])
X_test: pd.DataFrame = val_df.drop(columns=[TARGET_LABEL])

y_train: pd.Series = train_df[TARGET_LABEL]
y_test: pd.Series = val_df[TARGET_LABEL]

# Initialize the structured data regressor.
reg = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=100)

# Feed the structured data regressor with training data.
reg.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
