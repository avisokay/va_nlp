library(haven)

df = read_dta('ihme_child.dta')

head(df)

narratives = read.csv('phmrc_children_tokenized.csv')

narratives$tags

