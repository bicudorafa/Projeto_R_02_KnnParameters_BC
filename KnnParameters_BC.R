## Etapa 1 - Coletando os Dados

# Origem: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# Os dados do cancer da mama incluem 569 observa��es de biopsias de cancer 
# Cada observa��o possui 32 features, sendo a 1a delas sua identifica��o (ID), 
# a 2a � o diagnostico (diagnosis), sendo B = benigno e M = malicioso
# E as 30 features restantes s�o medidas laboratoriais nucleares cuja explicacao encontra-se no link
# http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

# link para os dados
link_dados <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

# definicao dos nomes das features
names_bc = c("id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
          "compactness_mean", "concavity_mean", "points_mean", "symmetry_mean", "dimension_mean", "radius_se",
          "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "points_se",
          "symmetry_se", "dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
          "smoothness_worst", "compactness_worst", "concavity_worst", "points_worst", "symmetry_worst","dimension_worst")

dados <- read.csv(link_dados, stringsAsFactors = F, col.names = names_bc)
str(dados)
head(dados)

## Etapa 2 - Explorando os Dados

# Excluindo a coluna ID

# Independete do m�todo de aprendizado de m�quina, colunas de identifica��o sempre devem ser exclu�das 
# Porque a feature ID serve unicamente para "prever" as observa��es de referencia
# Isso pode levar a overfitting ou outros problemas, por n�o possuir info �til (na maioria dos casos)
dados <- subset(dados, select = - id)
str(dados)
any(is.na(dados))

# Realizado o processo de Factoring em nossa vari�vel resposta (por boa parte dos algor�timos exigir)
dados$diagnosis <- factor(dados$diagnosis, levels = c('B', 'M'), labels = c('Benigno', 'Maligno'))

# Verificado a propor��o dos meus dados alvo
round(prop.table(table(dados$diagnosis))*100, digits = 1)

# Normaliza��o dos dados

# Outra caracter�stica comum a muitos processos de machine learning � a normaliza��o dos dados
# Diferen�as de escala entre os preditores podem sempre causar distor��es indesejadas 
# (ao menos, na maioria dos casos)

# fun��o base R para normaliza��o -> scale
dados_normalizados <- as.data.frame(scale(dados[2:31]))

# Fazendo uma compara��o entre algumas features antes e ap�s
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])
summary(dados_normalizados[c("radius_mean", "area_mean", "smoothness_mean")])

## Etapa 3: Treinando o modelo

# Carregando os pacotes necess�rios
# install.packages("class")
# install.packages("caTools")
library(caTools)
library(class)

# Criando os dados de treino e os de teste (obs.: neste dataset em especial n�o seria 
# necess�rio por ser randomizado originalmente)
set.seed(69)
amostra <- sample.split(dados$diagnosis, SplitRatio = 0.70)
dados_treino <- as.data.frame(subset(dados_normalizados, amostra == T))
dados_teste <- as.data.frame(subset(dados_normalizados, amostra == F))

# Criando os labels para identifica��o no modelo
dados_treino_labels <- subset(dados[1], amostra == T)[,1]
dados_teste_labels <- subset(dados[1], amostra == F)[,1]

# Criando o Modelo
modelo <- knn(train = dados_treino,
              test = dados_teste,
              cl = dados_treino_labels)

## Etapa 4: Avaliando e Interpretando o Modelo

# Carregando o gmodels
# install.packages("gmodels")
# install.packages("ggplot2")
library(gmodels)
library(ggplot2)

# Criando uma tabela cruzada dos dados previstos x dados atuais, ou seja, uma ConfusionMatrix e analisando taxa de erro
CrossTable(x = dados_teste_labels, y = modelo, prop.chisq = FALSE)
taxa_erro_inicial = mean(dados_teste_labels != modelo) 

# Calculando fun��o taxa de erro em rela��o ao tamanho do k
prev = NULL
taxa_erro = NULL
k_values = 1:25
#obs.: sempre que for realizar um loop, � bom costume come��-los vazios para garantir isso
suppressWarnings(
  for(i in k_values){
    set.seed(101)
    prev = knn(train = dados_treino,
               test = dados_teste,
               cl = dados_treino_labels,
               k = i)
    taxa_erro[i] = mean(dados_teste_labels != prev)
  })

df_erro <- data.frame(taxa_erro, k_values)
df_erro

# Plotando a rela��o entre as duas vari�veis
ggplot(df_erro, aes(x = k_values, y = taxa_erro)) + 
  geom_point()+ 
  geom_line(lty = "dotted", color = 'red') +
  labs(title = 'Taxa de Erro em Fun��o dos Valores de K',
       y = 'Taxa de Erro', x = 'Valores de K') +
  theme_classic()
  

