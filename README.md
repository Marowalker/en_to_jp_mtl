# en_to_jp_mtl

## **(THIS IS A WORK IN PROGRESS! DO NOT USE IT YET!)**

A prototype English to Japanese neural machine translator using a basic Seq2Seq model

The dataset used is the Japanese-English subtitle corpus, found here: https://www.kaggle.com/datasets/onslaught/japaneseenglish-subtitle-corpus

**IMPORTANT:** For Windows users, there will be an error when calling the TextVectorization layer's get_vocabulary() method. To fix, go to Settings > Time and Language > Language and region > Administrative language settings > Change system locale, then click on the option to use UTF-8 for worldwide language support. 