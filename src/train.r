# Required Libraries
library(jsonlite)
library(dplyr)
library(tidyr)
library(readr)
library(fastDummies)
library(randomForest)

# Define directories and paths

ROOT_DIR <- dirname(getwd())
MODEL_INPUTS_OUTPUTS <- file.path(ROOT_DIR, 'model_inputs_outputs')
INPUT_DIR <- file.path(MODEL_INPUTS_OUTPUTS, "inputs")
INPUT_SCHEMA_DIR <- file.path(INPUT_DIR, "schema")
DATA_DIR <- file.path(INPUT_DIR, "data")
TRAIN_DIR <- file.path(DATA_DIR, "training")
MODEL_ARTIFACTS_PATH <- file.path(MODEL_INPUTS_OUTPUTS, "model", "artifacts")
OHE_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'ohe.rds')
PREDICTOR_FILE_PATH <- file.path(MODEL_ARTIFACTS_PATH, "predictor", "predictor.rds")
IMPUTATION_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'imputation.rds')
LABEL_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'label_encoder.rds')
ENCODED_TARGET_FILE <- file.path(MODEL_ARTIFACTS_PATH, "encoded_target.rds")
TOP_3_CATEGORIES_MAP <- file.path(MODEL_ARTIFACTS_PATH, "top_3_map.rds")
COLNAME_MAPPING <- file.path(MODEL_ARTIFACTS_PATH, "colname_mapping.csv")


if (!dir.exists(MODEL_ARTIFACTS_PATH)) {
    dir.create(MODEL_ARTIFACTS_PATH, recursive = TRUE)
}
if (!dir.exists(file.path(MODEL_ARTIFACTS_PATH, "predictor"))) {
    dir.create(file.path(MODEL_ARTIFACTS_PATH, "predictor"))
}


# Reading the schema
# The schema contains metadata about the datasets. 
# We will use the schema to get information about the type of each feature (NUMERIC or CATEGORICAL)
# and the id and target features, this will be helpful in preprocessing stage.

file_name <- list.files(INPUT_SCHEMA_DIR, pattern = "*.json")[1]
schema <- fromJSON(file.path(INPUT_SCHEMA_DIR, file_name))
features <- schema$features

numeric_features <- features$name[features$dataType == "NUMERIC"]
categorical_features <- features$name[features$dataType == "CATEGORICAL"]
id_feature <- schema$id$name
target_feature <- schema$target$name
model_category <- schema$modelCategory


# Reading training data
file_name <- list.files(TRAIN_DIR, pattern = "*.csv")[1]
# Read the first line to get column names
header_line <- readLines(file.path(TRAIN_DIR, file_name), n = 1)
col_names <- unlist(strsplit(header_line, split = ",")) # assuming ',' is the delimiter
# Read the CSV with the exact column names
df <- read.csv(file.path(TRAIN_DIR, file_name), skip = 0, col.names = col_names, check.names=FALSE)

# Data Preprocessing
# Data preprocessing is very important before training the model, as the data may contain missing values in some cells. 
# Moreover, most of the learning algorithms cannot work with categorical data, thus the data has to be encoded.
# In this section we will impute the missing values and encode the categorical features. Afterwards the data will be ready to train the model.

# You can add your own preprocessing steps such as:

# Normalization
# Outlier removal
# Handling imbalanced classes
# Dropping or adding features

# Important note:
# Saving the values used for imputation during training step is crucial. 
# These values will be used to impute missing data in the testing set. 
# This is very important to avoid the well known problem of data leakage. 
# During testing, you should not make any assumptions about the data in hand, 
# alternatively anything needed during the testing phase should be learned from the training phase.
# This is why we are creating a dictionary of values used during training to reuse these values during testing.


# Impute missing data
imputation_values <- list()

columns_with_missing_values <- colnames(df)[apply(df, 2, anyNA)]
for (column in columns_with_missing_values) {
    if (column %in% numeric_features) {
        value <- median(df[, column], na.rm = TRUE)
    } else {
        value <- as.character(df[, column] %>% tidyr::replace_na())
        value <- value[1]
    }
    df[, column][is.na(df[, column])] <- value
    imputation_values[column] <- value
}
saveRDS(imputation_values, IMPUTATION_FILE)


# Encoding Categorical features

# The id column is just an identifier for the training example, so we will exclude it during the encoding phase.
# Target feature will be label encoded in the next step.

ids <- df[, id_feature]
target <- df[, target_feature]
df <- df %>% select(-all_of(c(id_feature, target_feature)))


# One Hot Encoding
if(length(categorical_features) > 0){
    top_3_map <- list()
    for(col in categorical_features) {
        # Get the top 3 categories for the column
        top_3_categories <- names(sort(table(df[[col]]), decreasing = TRUE)[1:3])

        # Save the top 3 categories for this column
        top_3_map[[col]] <- top_3_categories
        # Replace categories outside the top 3 with "Other"
        df[[col]][!(df[[col]] %in% top_3_categories)] <- "Other"
    }

    df_encoded <- dummy_cols(df, select_columns = categorical_features, remove_selected_columns = TRUE)
    encoded_columns <- setdiff(colnames(df_encoded), colnames(df))
    saveRDS(encoded_columns, OHE_ENCODER_FILE)
    saveRDS(top_3_map, TOP_3_CATEGORIES_MAP)
    df <- df_encoded
}



# for column names with special characters or spaces
sanitize_colnames <- function(names_vector) {
  # Check for unique original column names
  if (any(duplicated(names_vector))) {
    stop("Error: Given column names are not unique!")
  }

  # Trim spaces from column names
  names_vector <- trimws(names_vector)

  # Special characters sanitization
  sanitized_names <- gsub(" ", "_", names_vector)
  sanitized_names <- gsub("[^[:alnum:]_]", "_", sanitized_names)

  # Prefix with "feat_" - this is to get around columns that start with numbers
  sanitized_names <- paste0("feat_", sanitized_names)
  
  # Ensure uniqueness
  while(any(duplicated(sanitized_names))) {
    dupes <- table(sanitized_names)
    dupes <- as.character(names(dupes[dupes > 1]))
    
    for(d in dupes) {
      indices <- which(sanitized_names == d)
      sanitized_names[indices] <- paste0(d, "_", seq_along(indices))
    }
  }
  
  return(sanitized_names)
}

# save the column name mapping to a file
new_colnames <- sanitize_colnames(colnames(df))
colname_mapping <- data.frame(
  original = colnames(df),
  sanitized = new_colnames
)
write.csv(colname_mapping, COLNAME_MAPPING, row.names = FALSE)

# apply new column names to df
colnames(df) <- new_colnames


# Label encoding target feature
levels_target <- levels(factor(target))
encoded_target <- as.integer(factor(target, levels = levels_target)) - 1
saveRDS(levels_target, LABEL_ENCODER_FILE)
saveRDS(encoded_target, ENCODED_TARGET_FILE)


# Train the Classifier
# We choose Random Forest Classifier, but feel free to try your own and compare the results.
if (model_category == 'binary_classification'){
    # Use as.factor() for the target to ensure it's treated as classification
    model <- randomForest(as.factor(encoded_target) ~ ., data = df, ntree=100) 

} else if (model_category == "multiclass_classification") {
    # Same Random Forest function for multiclass
    model <- randomForest(as.factor(encoded_target) ~ ., data = df, ntree=100) 
}
saveRDS(model, PREDICTOR_FILE_PATH)
