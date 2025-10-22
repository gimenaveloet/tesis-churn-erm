# SCRIPT APARTADO 7 

# ==========================================================================================
  # Modelo predictivo para la Gestión del Riesgo Empresarial: Prevención de fuga de clientes en una Fintech
# ==========================================================================================
# 
# OBJETIVO: Desarrollar modelos de machine learning para predecir la probabilidad
#           de que un cliente abandone(churn) basándose en su 
#           comportamiento histórico de transacciones.
#
# METODOLOGÍA: 
#   1. Definición y validación de churn
#   2. Ingeniería de características (feature engineering)
#   3. Análisis exploratorio de datos 
#   4. Desarrollo de tres modelos: Árbol de Decisión, Random Forest, Regresión Logística
#   5. Comparación y selección del mejor modelo
#


# 1. LIBRERÍAS Y PAQUETES --------------------------------------------------
paquetes <- c(
  "DBI","odbc","tidyverse","caret","corrplot","arrow","randomForest","GGally",
  "e1071","reshape2","RColorBrewer","ggthemes","kernlab","magrittr","knitr",
  "mlr","ROSE","MLmetrics","scales","rpart","rpart.plot",
  "here","fs","glmnet","car","lubridate","ggplot2","dplyr","lubridate"
)

# Configuración del repositorio CRAN para instalación de paquetes
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Instalar sólo lo que falte 
instalados <- rownames(installed.packages())
to_install <- setdiff(paquetes, instalados)
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

# Cargar en orden 
to_load <- paquetes
suppressPackageStartupMessages(invisible(lapply(
  to_load, library, character.only = TRUE
)))


# - tidyverse: Manipulación y visualización de datos (gramática de datos)
# - caret: Machine learning unificado (entrenamiento, validación cruzada, métricas)
# - randomForest: Algoritmo de ensemble learning (bagging)
# - rpart: Árboles de decisión con poda automática
# - glmnet: Regresión logística con regularización (LASSO/Ridge)
# - pROC: Análisis de curvas ROC y métricas de clasificación
# - ROSE: Técnicas de balanceo de clases (oversampling/undersampling)


# 2. EXTRACCIÓN DE DATOS -----------------------------------------------------

# La extracción de datos es el primer paso del proceso KDD (Knowledge Discovery in Databases)
# Se utilizan archivos Parquet para garantizar portabilidad y eficiencia:
# - Formato columnar optimizado para análisis
# - Compresión eficiente
# - Compatibilidad multiplataforma


### A fin de poder compartir el script y que pueda ser utilizado por cualquier persona, el dataset se guardó en un archivo .parquet.  
ventas <- read_parquet("ventas.parquet")

# Transacciones (ventas) 
# - cliente_id: Identificador único del cliente
# - fecha_venta: Timestamp de la transacción (para análisis temporal)
# - producto: Tipo de producto utilizado
# - estado: Estado de la transacción (aprobada/rechazada)
# - monto_venta: Valor monetario de la transacción


cuentas <- read_parquet("cuentas.parquet")

# -cliente_id: Identificador único del cliente
# -province: Provincia en la que está ubicado el cliente
# -tipo_cuenta: Profesional, Ocasional o Compañia
# -rubro: Sector en el que opera el cliente (es un id)


  # 2.1 Conexión SQL Server
# con <- dbConnect(
#   odbc::odbc(),
#   Driver   = "SQL Server",     
#   Server   = "#####",  
#   Database = "####", 
#   UID      = "#####",      
#   PWD      = "#############",    
#   Port     = 1433
# )

  # 2.2 Transacciones (ventas)
# query_ventas <- "
# SELECT  account_id cliente_id,  
#         date fecha_venta, 
#         product producto,
#         status_normalizado estado,
#         Amount_USD monto_venta 
# FROM Fact_Transactions a
# LEFT JOIN dim_responses b on a.idresponse=b.IDResponse
# LEFT JOIN dim_projects c on a.idproject=c.IDProject
# LEFT JOIN dim_dates d on a.IDDate = d.IDDate 
# LEFT JOIN DIM_ORIGINS e on a.IDOrigin=e.IDOrigin
# WHERE  a.idpartner = 18 and Type_Normalizado='SALE'
# "
# ventas <- dbGetQuery(con, query_ventas)

# Verifico estructura
# glimpse(ventas)

  # 2.3 Datos de cuentas (perfil de clientes)
# query_cuentas <- "
# SELECT DISTINCT a.account_id cliente_id,
#                 c.province,
#                 account_type tipo_cuenta, 
#                 classification_id rubro 
# FROM Account_Description a 
# LEFT JOIN Subsidiary_Description b on a.account_id=b.account_id and a.IDPartner=b.IDPartner 
# LEFT JOIN address_description c on b.address_id=c.id_address and b.idpartner=c.idpartner 
# WHERE a.idpartner = 18 
# "
# cuentas <- dbGetQuery(con, query_cuentas)

# Verifico estructura
# glimpse(cuentas)

#Cierro conexión
# dbDisconnect(con)

# Las secciones 2.2 y 2.3 se corresponden a la ejecución original desde SQL Server. No aplica para reproducibilidad. 

  # 2.4 Limpieza inicial
# La limpieza de datos es fundamental para la calidad del análisis:
# - Imputación de valores faltantes con valores por defecto lógicos
# - Conversión de tipos de datos para análisis posterior
# - Validación de consistencia de datos


ventas <- ventas %>%
  mutate(producto = ifelse(is.na(producto), "mpos", producto),   ###No contamos con el dato para las tx viejas, pero solo existía mpos.
         fecha_venta = as.Date(fecha_venta))

cuentas <- cuentas %>%
  mutate(province = ifelse(is.na(province), "Ciudad Autónoma de Buenos Aires", province))  ### Por default es CABA por el Partner


# 3. DEFINICIÓN Y VALIDACIÓN DE CHURN ----------------------------------------
#
# La definición de churn es crítica para el éxito del modelo. Se evalúan
# múltiples definiciones basadas en:
# 1. Frecuencia de transacciones (híbrido)
# 2. Períodos de inactividad fijos
# 3. Validación con datos futuros (holdout temporal)

# Definir fecha de corte
fecha_inicio <- as.Date("2023-06-01") #No tiene mucho sentido analizar tx tan antiguas para la definición.
fecha_corte <- as.Date("2025-02-28") #Tengo que tener un margen para evualuar si fue churn o no realmente.

# Filtrar ventas hasta la fecha de corte
ventas_corte <- ventas %>%
  filter(fecha_venta <= fecha_corte) %>%
  filter(fecha_venta >= fecha_inicio)

# 3.1 Calculo de frecuencia por cliente
# Las métricas de frecuencia son fundamentales para detectar patrones de churn:
# - Frecuencia promedio: Tiempo promedio entre transacciones
# - Días desde última venta: Indicador de actividad reciente
# - Variabilidad: Consistencia en el comportamiento del cliente

frecuencias <- ventas_corte %>%
  arrange(cliente_id, fecha_venta) %>%
  group_by(cliente_id) %>%
  summarise(
    cantidad_ventas = n(),
    primera_venta = min(fecha_venta),
    ultima_venta = max(fecha_venta),
    frecuencia_prom = ifelse(n() > 1,
                             as.numeric((max(fecha_venta) - min(fecha_venta)) / (n() - 1)),
                             NA),
    dias_desde_ultima_venta = as.numeric(fecha_corte - ultima_venta),
    .groups = "drop"
  )

# Definiciones de churn evaluadas
# Se evalúan tres definiciones de churn para encontrar la más efectiva:

# Definición 1: Híbrido frecuencia × 3
# Un cliente es considerado churn si no ha transaccionado en un período
# superior a 3 veces su frecuencia promedio histórica

frecuencias <- frecuencias %>%
  mutate(
    umbral_fuga_f3 = frecuencia_prom * 3,
    fugado_f3 = ifelse(dias_desde_ultima_venta > umbral_fuga_f3, 1, 0)
  )

# Definición 2: Híbrido frecuencia × 2
# Versión más estricta que la anterior (2x en lugar de 3x)
frecuencias <- frecuencias %>%
  mutate(
    umbral_fuga_f2 = frecuencia_prom * 2,
    fugado_f2 = ifelse(dias_desde_ultima_venta > umbral_fuga_f2, 1, 0)
  )

# Definición 3: 60 días sin actividad desde última venta
# Definición fija basada en reglas de negocio estándar
frecuencias <- frecuencias %>%
  mutate(
    fugado_60d = ifelse(dias_desde_ultima_venta > 60, 1, 0)
  )

# 3.2 Validación con ventas reales post-corte
# La validación temporal es crucial para evaluar la efectividad real
# de las definiciones de churn. Se utilizan datos posteriores al corte
# para verificar si los clientes marcados como churn realmente no volvieron.

ventas_validacion <- ventas %>%
  filter(fecha_venta > fecha_corte)

clientes_que_volvieron <- ventas_validacion %>%
  distinct(cliente_id) %>%
  mutate(volvio_a_vender = 1)

# Función para evaluar una definición de fuga
evaluar_definicion <- function(df, columna_fugado, nombre_def) {
  df %>%
    select(cliente_id, fugado = {{columna_fugado}}) %>%
    left_join(clientes_que_volvieron, by = "cliente_id") %>%
    mutate(
      volvio_a_vender = ifelse(is.na(volvio_a_vender), 0, volvio_a_vender),
      acierto = case_when(
        fugado == 1 & volvio_a_vender == 0 ~ "Acertó (fugado)",
        fugado == 1 & volvio_a_vender == 1 ~ "Falló (volvió)",
        fugado == 0 & volvio_a_vender == 1 ~ "Acertó (activo)",
        fugado == 0 & volvio_a_vender == 0 ~ "Falló (se fue)"
      )
    ) %>%
    count(acierto, name = "n") %>%
    mutate(
      pct = round(n / sum(n) * 100, 1),
      definicion = nombre_def
    )
}

# Evaluo cada definición
eval_f3 <- evaluar_definicion(frecuencias, fugado_f3, "Frecuencia × 3")
eval_f2 <- evaluar_definicion(frecuencias, fugado_f2, "Frecuencia × 2")
eval_60 <- evaluar_definicion(frecuencias, fugado_60d, "Inactividad > 60 días")

# Consolido resultados
comparacion <- bind_rows(eval_f3, eval_f2, eval_60)

# 3.3 Resultado: tasa de aciertos y errores
comparacion %>%
  pivot_wider(names_from = definicion, values_from = c(n, pct), values_fill = 0)

# CONCLUSIÓN: Se selecciona la definición ">60 días de inactividad" por su
# mejor balance entre precisión y simplicidad interpretativa.


# 4. GENERACIÓN DE FEATURES --------------------------------------------------
#
# La generación de features es crucial para el rendimiento del modelo.
# Se generan features en tres categorías:
# 1. Features históricos: Comportamiento a largo plazo del cliente
# 2. Features temporales: Patrones de actividad en ventanas específicas
# 3. Features de contexto: Información demográfica y de perfil

ventana_churn  <- 60

  # 4.1 Agrergo producto, día, aprobación
# Filtrar ventas hasta la fecha de corte
ventas_target <- ventas %>%
  filter(fecha_venta <= fecha_corte) 

# Filtro de cuentas activas en los últimos 60 días
# Solo analizamos clientes que estuvieron activos recientemente
# para evitar sesgos por clientes inactivos históricamente

cuentas_activas <- ventas %>%
  filter(fecha_venta > (fecha_corte - ventana_churn) & fecha_venta <= fecha_corte) %>%
  distinct(cliente_id) 

# Generación de features derivados
ventas_target <- ventas_target %>%
  mutate(
    dia_semana = wday(fecha_venta, label = FALSE, week_start = 1),   # Lunes = 1
    canal = case_when(
      producto %in% c("mpos", "standalone", "qr", "SDK", "smartpos", "softpos", "caja_pos") ~ "presencial",
      producto %in% c("payment_link", "api_checkout", "plugins", "online_catalog", "api_payments", "PCT", "tienda_geo") ~ "online",
      TRUE ~ "otro"
    ),
    aprobada = ifelse(estado == "APPROVED", 1, 0)
  )

  # 4.2 Features históricos por cliente
# Los features históricos capturan el comportamiento a largo plazo
# del cliente, incluyendo patrones de frecuencia, valor y diversidad

features <- ventas_target %>%
  semi_join(cuentas_activas, by = "cliente_id") %>%              
  arrange(cliente_id, fecha_venta) %>%
  group_by(cliente_id) %>%
  summarise(
    dias_entre_ultimas_2_ventas = ifelse(n() >= 2, as.numeric(diff(tail(fecha_venta, 2))), NA),
    variabilidad_entre_ventas   = ifelse(n() >= 2, sd(as.numeric(diff(sort(fecha_venta)))), NA),
    frecuencia_promedio_historica = ifelse(n() > 1,
                                           as.numeric((max(fecha_venta) - min(fecha_venta)) / (n() - 1)), NA),
    ticket_promedio         = mean(monto_venta, na.rm = TRUE),
    desviacion_ticket       = sd(monto_venta, na.rm = TRUE),
    monto_total_historico   = sum(monto_venta, na.rm = TRUE),
    mes_ultima_venta        = lubridate::month(max(fecha_venta)),
    ventas_en_semana        = sum(!dia_semana %in% c(6, 7)),     
    ventas_en_fin_de_semana = sum(dia_semana %in% c(6, 7)),
    q_productos             = n_distinct(producto),
    ventas_online           = any(canal == "online"),
    ventas_presencial       = any(canal == "presencial"),
    ventas_mixtas           = ventas_online & ventas_presencial,
    total_ventas            = n(),
    total_aprobadas         = sum(aprobada),
    tasa_aprobacion         = ifelse(total_ventas > 0, total_aprobadas / total_ventas, NA),
    usuario_nuevo           = min(fecha_venta) >= (fecha_corte - 90),
    .groups = "drop"
  )

# ventas_en_semana = sum(!dia_semana %in% c("6","7")),
 #   ventas_en_fin_de_semana = sum(dia_semana %in% c("6","7")),

  # 4.3 Features de Ventanas temporales
# Las ventanas temporales capturan patrones de comportamiento reciente
# que pueden ser indicadores tempranos de churn. Se evalúan múltiples ventanas
# para capturar diferentes escalas temporales.

ventanas <- function(d) {
  ventas_target %>%
    filter(fecha_venta > (fecha_corte - d), fecha_venta <= fecha_corte) %>%
    group_by(cliente_id) %>%
    summarise(
      !!paste0("q_ventas_ultimos_", d, "d")      := n(),
      !!paste0("monto_total_ultimos_", d, "d")   := sum(monto_venta, na.rm = TRUE),
      !!paste0("ticket_prom_ultimos_", d, "d")   := mean(monto_venta, na.rm = TRUE),
      .groups = "drop"
    )
}

# Generación de features para múltiples ventanas temporales
ventanas_15 <- ventanas(15)
ventanas_30 <- ventanas(30)
ventanas_60 <- ventanas(60)
ventanas_90 <- ventanas(90)


# La consolidación combina todos los features en un dataset unificado
# para el entrenamiento del modelo, incluyendo imputación de valores faltantes
# y generación de features derivados.

dataset_churn <- features %>%
  left_join(ventanas_15, by = "cliente_id") %>%
  left_join(ventanas_30, by = "cliente_id") %>%
  left_join(ventanas_60, by = "cliente_id") %>%
  left_join(ventanas_90, by = "cliente_id") %>%
  mutate(
    q_ventas_ultimos_15d = replace_na(q_ventas_ultimos_15d, 0),
    q_ventas_ultimos_30d = replace_na(q_ventas_ultimos_30d, 0),
    q_ventas_ultimos_60d = replace_na(q_ventas_ultimos_60d, 0),
    q_ventas_ultimos_90d = replace_na(q_ventas_ultimos_90d, 0),
    ticket_prom_ultimos_15d = replace_na(ticket_prom_ultimos_15d, 0),
    ticket_prom_ultimos_30d = replace_na(ticket_prom_ultimos_30d, 0),
    ticket_prom_ultimos_60d = replace_na(ticket_prom_ultimos_60d, 0),
    ticket_prom_ultimos_90d = replace_na(ticket_prom_ultimos_90d, 0),
    tasa_cambio_actividad = q_ventas_ultimos_15d / pmax(q_ventas_ultimos_30d / 2, 1),
    variacion_frecuencia  = (15 / pmax(q_ventas_ultimos_15d, 1)) - frecuencia_promedio_historica,
    variacion_ticket      = (ticket_prom_ultimos_15d - ticket_promedio) / ticket_promedio

  ) %>%
  select(-total_ventas, -total_aprobadas)

  # 4.4 Unión con contexto
# La información demográfica y de perfil proporciona contexto adicional
# para la predicción de churn, permitiendo segmentación y análisis más granular.

dataset_churn <- dataset_churn %>%
  semi_join(cuentas_activas, by = "cliente_id")   

dataset_final <- dataset_churn %>%
  left_join(cuentas, by = "cliente_id") %>%
  rename(provincia = province) %>%                
  select(-matches("^province$"))  #drop duplicado si hay

# 4.5 Generación de variable target
# =============================================================================
# La variable target se genera utilizando la definición de churn seleccionada
# (>60 días de inactividad) y se valida con datos futuros.

# Datos de validación para generación de target

ventas_post_corte <- ventas %>%
  filter(fecha_venta > fecha_corte, fecha_venta <= (fecha_corte + ventana_churn))

clientes_que_volvieron <- ventas_post_corte %>%
  distinct(cliente_id) %>%
  mutate(fugado = 0)

# Generación de variable target final
dataset_final <- dataset_final %>%
  left_join(clientes_que_volvieron, by = "cliente_id") %>%
  mutate(fugado = ifelse(is.na(fugado), 1, fugado)) 


# 5. ANÁLISIS EXPLORATORIO DE DATOS  (KDD) ---------------------------------------------------------------------
#
# Esta etapa es fundamental para entender la estructura de los datos,
# identificar patrones, detectar outliers y validar la calidad de los features
# antes del modelado

# 5.1. Análisis exploratorio inicial
glimpse(dataset_final)

  # 5.2 Distribución target
# El análisis de la distribución de la variable target es crucial para:
# - Identificar desbalance de clases
# - Decidir estrategias de balanceo
# - Interpretar métricas de evaluación

cat("Distribución de la variable 'fugado':\n")
print(table(dataset_final$fugado))
print(prop.table(table(dataset_final$fugado)))

# CONCLUSIÓN: Fuerte desbalance de clases (17.6% churn, 82.4% activos)

  # 5.3. Detección y tratamiento de valores faltantes
# Los valores faltantes pueden afectar significativamente el rendimiento
# del modelo. Se implementa una estrategia de imputación diferenciada por tipo de variable.

na_summary <- sapply(dataset_final, function(x) sum(is.na(x)))
na_summary <- sort(na_summary[na_summary > 0], decreasing = TRUE)

print(na_summary)


# 5.3.1 Estrategia de imputación diferenciada
# =============================================================================
# Diferentes tipos de variables requieren diferentes estrategias de imputación:

# A) Variables de ventanas temporales: NA = no hubo ventas en la ventana → imputar 0

vars_ventanas <- c(
  "q_ventas_ultimos_15d","monto_total_ultimos_15d","ticket_prom_ultimos_15d",
  "q_ventas_ultimos_30d","monto_total_ultimos_30d","ticket_prom_ultimos_30d",
  "q_ventas_ultimos_60d","monto_total_ultimos_60d","ticket_prom_ultimos_60d",
  "q_ventas_ultimos_90d","monto_total_ultimos_90d","ticket_prom_ultimos_90d"
)
vars_ventanas <- intersect(vars_ventanas, names(dataset_final))  

dataset_final <- dataset_final %>%
  mutate(across(all_of(vars_ventanas), ~ tidyr::replace_na(., 0)))

# B) Variables que requieren mínimo 2 eventos para calcularse:
#    NA = no se puede calcular → usar -1 como centinela explícito

vars_dep_2_eventos <- c("variabilidad_entre_ventas",
                        "dias_entre_ultimas_2_ventas",
                        "frecuencia_promedio_historica",
                        "desviacion_ticket",
                        "variacion_frecuencia",
                        "variacion_ticket"
                        )
vars_dep_2_eventos <- intersect(vars_dep_2_eventos, names(dataset_final))

dataset_final <- dataset_final %>%
  mutate(across(all_of(vars_dep_2_eventos), ~ tidyr::replace_na(., -1)))

# C) Imputación residual con mediana para casos puntuales
dataset_final <- dataset_final %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))


# Verificación de NAs remanentes
na_summary_after <- sapply(dataset_final, function(x) sum(is.na(x)))
na_summary_after <- sort(na_summary_after[na_summary_after > 0], decreasing = TRUE)
print(na_summary_after)


  # 5.4 Revisión y transformación de tipos de variables
# La transformación correcta de tipos de variables es esencial para:
# - Algoritmos específicos (algunos requieren factores, otros numéricos)
# - Interpretación correcta de resultados
# - Evitar errores en el modelado

summary(dataset_final)

    # 5.4.1 Convertir flags a lógico si están como numéricos
dataset_final <- dataset_final %>%
  mutate(across(c(ventas_online, ventas_presencial, ventas_mixtas, usuario_nuevo), as.logical))

    # 5.4.2 Convertir categóricas a factor  ## Para algunos modelos será necesario transformarlos en numericos.
dataset_final <- dataset_final %>%
  mutate(
    tipo_cuenta = as.factor(tipo_cuenta),
    provincia = as.factor(provincia)
  )

  # 5.5 Matriz de correlación entre numéricas
# El análisis de correlaciones ayuda a:
# - Identificar variables redundantes (multicolinealidad)
# - Detectar patrones de dependencia
# - Guiar la selección de features

numeric_vars <- dataset_final %>%
  select(where(is.numeric), -cliente_id, -fugado)

cor_matrix <- cor(numeric_vars, use = "complete.obs")

corrplot::corrplot(cor_matrix, method = "color", tl.cex = 0.7, number.cex = 0.6, type = "upper")

# 5.6 Histogramas
# El análisis de distribuciones es crucial para:
# - Identificar outliers
# - Detectar sesgos en los datos
# - Decidir transformaciones necesarias

num_vars <- dataset_final %>%
  select(where(is.numeric), -cliente_id, -fugado)

num_vars <- num_vars %>%
  mutate(across(everything(), ~ as.numeric(.)))

# Visualización de distribuciones
num_vars %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(x = value)) +
  facet_wrap(~ name, scales = "free", ncol = 4) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  theme_minimal()


# 6. PROCESAMIENTO FINAL Y PARTICIÓN DEL CONJUNTO DE DATOS --------------------------------------
#
# La partición de datos es fundamental para la validación del modelo:
# - Conjunto de entrenamiento: Para ajustar los parámetros del modelo
# - Conjunto de prueba: Para evaluar el rendimiento final (no visto durante entrenamiento)
# - Estratificación: Para mantener proporciones de clases en ambos conjuntos

 # 6.1 Preparación de datos
# Conversión de variable target a factor para algoritmos de clasificación
dataset_final$fugado[dataset_final$fugado==0] <- 'No' 
dataset_final$fugado[dataset_final$fugado==1] <- 'Si' 

# Conversión de variables categóricas a factor
dataset_final[sapply(dataset_final, is.character)] <- lapply(dataset_final[sapply(dataset_final, is.character)], 
                                           as.factor)

dataset_final$fugado <- as.factor(dataset_final$fugado)

str(dataset_final)


  #6.2 Data partition (Train and Test)

# La partición estratificada garantiza que las proporciones de clases
# se mantengan similares en entrenamiento y prueba, evitando sesgos.

# Configuración de semilla para reproducibilidad

set.seed(123)

    # Partición estratificada 70% entrenamiento, 30% testeo
train_index <- createDataPartition(dataset_final$fugado, p = 0.7, list = FALSE)

    # Creo datasets de entrenamiento y testeo
train_data <- dataset_final[train_index, ]
test_data  <- dataset_final[-train_index, ]

    # Verifico proporciones antes de balancear
cat("Proporciones antes del oversampling (entrenamiento):\n")
print(prop.table(table(train_data$fugado)))


  #6.3 Balanceo - Oversampling del set de entrenamiento (solo train_data)
# El desbalance de clases puede causar que el modelo se sesgue hacia
# la clase mayoritaria. El oversampling duplica ejemplos de la clase minoritaria
# para equilibrar las proporciones.

# Oversampling del conjunto de entrenamiento
set.seed(123)  # para reproducibilidad también en ovun.sample
n_mayoria <- max(table(train_data$fugado))
train_data_bal <- ovun.sample(fugado ~ ., data = train_data, method = "over", N = n_mayoria * 2)$data

    # Verifico proporciones después de balancear
cat("Proporciones después del oversampling (entrenamiento balanceado):\n")
print(prop.table(table(train_data_bal$fugado)))

    # El set de testeo queda como el original y representa la distribución real
cat("Proporciones en test (sin balancear):\n")
print(prop.table(table(test_data$fugado)))



# 7. Desarrollo de modelo: ÁRBOL DE DECISIÓN ------------------------------
#
# Los árboles de decisión son modelos interpretables que:
# - Dividen recursivamente el espacio de características
# - Utilizan criterios de división (Gini, entropía) para maximizar la pureza
# - Aplican poda para evitar sobreajuste
# - Proporcionan reglas interpretables para la toma de decisiones

set.seed(123)

# Preparación de datos para árbol de decisión
datos  <- train_data_bal %>% dplyr::select(-cliente_id)
target <- "fugado"  # factor con niveles: c("No","Si")

# 7.1 Parámetros del árbol
# Los parámetros del árbol controlan su crecimiento y complejidad:
# - cp: Parámetro de complejidad (menor = árbol más complejo)
# - minsplit: Mínimo de observaciones para dividir un nodo
# - minbucket: Mínimo de observaciones en un nodo terminal
# - maxdepth: Profundidad máxima del árbol

ctrl <- rpart.control(
  cp = 1e-4,       # cp bajo para que crezca grande y genere la secuencia
  minsplit = 15,
  minbucket = 7,
  maxdepth = 30,
  xval = 0        # No necesito CV interna para llenar cptable - Hago validación cruzada externa con cv_externa_por_cp
)
split_metric <- "gini"  # o "information" (entropía)

# 7.2 Funciones 
# Las funciones auxiliares implementan la metodología de validación cruzada
# externa para selección del parámetro de complejidad óptimo.

# Función para entrenar árbol completo
fit_full <- function(df) {
  rpart(
    as.formula(paste(target, "~ .")),
    data = df,
    method = "class",
    parms = list(split = split_metric),
    control = ctrl
  )
}

# Función de validación cruzada externa por parámetro cp
cv_externa_por_cp <- function(df, cp_values, k = 5) {
  folds <- createFolds(df[[target]], k = k, list = TRUE)
  res <- data.frame(cp = cp_values, error_promedio = NA_real_, error_sd = NA_real_)

  for (i in seq_along(cp_values)) {
    cp_i <- cp_values[i]
    errs <- c()
    for (kidx in seq_along(folds)) {
      valid_idx  <- folds[[kidx]]
      df_train <- df[-valid_idx, , drop = FALSE]
      df_valid <- df[ valid_idx, , drop = FALSE]
      
      t_full <- fit_full(df_train)
      t_prun <- prune(t_full, cp = cp_i)
      
      pred <- predict(t_prun, newdata = df_valid, type = "class")
      errs <- c(errs, mean(pred != df_valid[[target]]))
    }
    res$error_promedio[i] <- mean(errs)
    res$error_sd[i]       <- sd(errs)
  }
  res
}

# Función para limitar número de hojas
prune_to_leaves_limit <- function(tree_full, cpt, max_leaves = Inf) {
  # Recorre CP de mayor→menor (árbol más simple→más grande)
  for (cp_try in rev(cpt$CP)) {
    tr <- prune(tree_full, cp = cp_try)
    if (sum(tr$frame$var == "<leaf>") <= max_leaves) {
      return(list(tree = tr, cp = cp_try))
    }
  }
  list(tree = prune(tree_full, cp = min(cpt$CP)), cp = min(cpt$CP))
}
  

# 7.3 Creo arbol grande
# Se entrena un árbol completo para generar la secuencia de parámetros
# de complejidad que luego se evalúan mediante validación cruzada.

arbol_completo <- fit_full(datos)
cpt <- as.data.frame(printcp(arbol_completo))   # columnas: CP, nsplit, rel error, xerror, xstd
cp_values <- cpt$CP

# 7.4 Aplicar validación cruzada externa para seleccionar el α (cp) óptimo
# La validación cruzada externa proporciona una estimación no sesgada
# del rendimiento del modelo para diferentes niveles de complejidad.

cvext <- cv_externa_por_cp(datos, cp_values, k = 5)
cp_elegido <- cvext$cp[ which.min(cvext$error_promedio) ]

cat(sprintf("cp seleccionado por CV externa: %.6g\n", cp_elegido))


# 7.5 Opcional limitar ramas.
# Limitar el número de hojas puede mejorar la interpretabilidad
# sin sacrificar significativamente el rendimiento.

max_hojas <- 25
ajuste_lim <- prune_to_leaves_limit(arbol_completo, cpt, max_leaves = max_hojas)
# nos quedamos con el mayor cp entre el elegido y el necesario para cumplir el límite
cp_elegido <- max(cp_elegido, ajuste_lim$cp)

# 7.6 Generación del árbol final
arbol_final <- prune(arbol_completo, cp = cp_elegido)

cat(sprintf("Árbol final: cp=%.6g | hojas=%d\n",
            cp_elegido, sum(arbol_final$frame$var == "<leaf>")))

# 7.7 Visualización
# La visualización del árbol es crucial para la interpretabilidad
# y comprensión de las reglas de decisión.

rpart.plot(
  arbol_final,
  main = paste0("Árbol final (cp=", signif(cp_elegido, 4),
                ", hojas=", sum(arbol_final$frame$var == "<leaf>"), ")"),
  cex = 0.6,
  fallen.leaves = TRUE
)

 rpart.rules(arbol_final, cover = TRUE, roundint = FALSE)


# 7.8 Evaluación test
 # La evaluación en datos no vistos proporciona una estimación realista
 # del rendimiento del modelo en producción.
 
 # Predicciones en conjunto de prueba
 
pred_arbol <- predict(arbol_final, newdata = test_data %>% dplyr::select(-cliente_id), type = "class")
pred_prob  <- predict(arbol_final, newdata = test_data %>% dplyr::select(-cliente_id), type = "prob")[, "Si"]

# Matriz de confusión
conf_matrix <- caret::confusionMatrix(pred_arbol, test_data[[target]], positive = "Si")
print(conf_matrix)


# 7.9 Métricas adicionales 
# Las métricas F1 proporcionan un balance entre precisión y sensibilidad,
# especialmente importante en problemas de clasificación desbalanceada.

# F1-Score para clase "Si" (churn)
f1_si = 2 * (conf_matrix$byClass["Pos Pred Value"] * conf_matrix$byClass["Sensitivity"]) /
  (conf_matrix$byClass["Pos Pred Value"] + conf_matrix$byClass["Sensitivity"])
# F1-Score para clase "No" (no churn)
f1_no = 2 * (conf_matrix$byClass["Neg Pred Value"] * conf_matrix$byClass["Specificity"]) /
  (conf_matrix$byClass["Neg Pred Value"] + conf_matrix$byClass["Specificity"])


# AUC
if (requireNamespace("pROC", quietly = TRUE)) {
  roc_obj <- pROC::roc(response = test_data[[target]], predictor = pred_prob, levels = c("No", "Si"))
  auc_val <- as.numeric(pROC::auc(roc_obj))
} else {
  auc_val <- NA_real_
}


# 7.10 Guardar métricas en df
metricas_modelos <- data.frame(
  modelo = "Árbol de decisión (CV externa)",
  accuracy = conf_matrix$overall["Accuracy"],
  error_rate = 1 - conf_matrix$overall["Accuracy"],
  kappa = conf_matrix$overall["Kappa"],
  sensibilidad = conf_matrix$byClass["Sensitivity"],       # recall de "Si"
  especificidad = conf_matrix$byClass["Specificity"],
  precision_si= conf_matrix$byClass["Pos Pred Value"],
  f1_no = f1_no,
  f1_si = f1_si,
  balanced_accuracy = conf_matrix$byClass["Balanced Accuracy"],
  auc = auc_val,
  row.names = NULL
)
print(metricas_modelos)


# 7.11 Análisis de importancia de variables del árbol
# La importancia de variables en árboles de decisión se basa en la
# reducción total de impureza (Gini) que cada variable contribuye.


# Extraer importancia de variables del árbol
tree_importance <- arbol_final$variable.importance
tree_importance_df <- data.frame(
  variable = names(tree_importance),
  importance = as.numeric(tree_importance)
) %>% 
  arrange(desc(importance))

print("Top 10 variables más importantes del árbol:")
print(head(tree_importance_df, 10))

# Visualización de importancia del árbol
tree_importance_plot <- tree_importance_df %>%
  head(15) %>%
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "Top 15 Variables más Importantes - Árbol de Decisión",
       x = "Variable", 
       y = "Importancia") +
  theme_minimal()

print(tree_importance_plot)


# 7.12 Guardo el modelo 
# La persistencia del modelo incluye no solo el objeto entrenado,
# sino también metadatos necesarios para reproducibilidad y despliegue.

# Carpeta destino
dir_create(here("artifacts"))

# Bundle
tree_artifact <- list(
  model_type      = "rpart",
  model           = arbol_final,
  # objetos útiles para reproducibilidad / explicación
  cp_selected     = cp_elegido,
  cptable         = cpt,                # tabla CP del árbol grande
  split_metric    = split_metric,
  control_params  = ctrl,
  # esquema de datos
  feature_names   = colnames(datos %>% dplyr::select(-all_of(target))),
  target_name     = target,
  target_levels   = levels(train_data[[target]]),
  # métricas en test
  metrics = list(
    confusion      = conf_matrix,
    f1_no          = as.numeric(f1_no),
    f1_si          = as.numeric(f1_si),
    auc            = as.numeric(auc_val)
  ),
  # housekeeping
  seed            = 123,
  session         = sessionInfo(),
  created_at      = Sys.time()
)

saveRDS(tree_artifact, here("artifacts", "tree_bundle_v1.rds"), compress = "xz")


# 8. Desarrollo de modelo: RANDOM FOREST ------------------------------
#
# Random Forest es un algoritmo de ensemble learning que combina múltiples
# árboles de decisión para mejorar la precisión y reducir el sobreajuste:
# - Bagging (Bootstrap Aggregating): Entrena múltiples árboles en muestras bootstrap
# - Selección aleatoria de variables: En cada división, considera solo un subconjunto aleatorio
# - Agregación por votación: La predicción final es la moda de las predicciones individuales
# - Ventajas: Manejo automático de overfitting, importancia de variables, robustez a outliers

set.seed(123)

# Preparación de datos para Random Forest
datos_rf <- train_data_bal %>% dplyr::select(-cliente_id)
target_rf <- "fugado"

# 8.1 Definición del espacio de hiperparámetros para Grid Search
# La optimización de hiperparámetros es crucial para el rendimiento del modelo.
# Se utiliza Grid Search con validación cruzada para encontrar la combinación óptima:
# - mtry: Número de variables consideradas en cada división (regla: √M donde M = total variables)
# - ntree: Número de árboles en el ensemble (más árboles = mejor estabilidad, mayor costo computacional)
# - sampsize: Tamaño de muestra para cada árbol (bootstrap sampling)
# - nodesize: Tamaño mínimo de nodos terminales (controla complejidad)
# - maxnodes: Número máximo de nodos terminales (controla profundidad)

# Cálculo del mtry por defecto según teoría
n_predictores <- ncol(datos_rf) - 1  # -1 porque una columna es el target
mtry_default <- round(sqrt(n_predictores))

# Configuración hiperparámetros principales (defino grid de hiperparámetros para tuning)
grid_rf <- expand.grid(
  mtry = c(max(1, mtry_default - 2), mtry_default, min(n_predictores, mtry_default + 2)),
  ntree = c(100, 300, 500),
  sampsize = c(0.6, 0.7, 0.8, 1), # 1= tamaño N (con reemplazo), lo que en promedio deja ~63% únicas (bootstrap clásico).
  nodesize = c(1, 5, 10),
  maxnodes = c(NA, 50, 100)
)

# Filtrar combinaciones válidas (mtry no puede ser mayor que el número de predictores)
grid_rf <- grid_rf[grid_rf$mtry <= n_predictores, ]
cat("Total de combinaciones a probar:", nrow(grid_rf), "\n") #324


# 8.2 Función para entrenar y evaluar Random Forest con validación cruzada
# La validación cruzada proporciona una estimación no sesgada del rendimiento
# del modelo para diferentes combinaciones de hiperparámetros, evitando overfitting
# en la selección de parámetros.

train_rf_cv <- function(data, target, mtry, ntree, sampsize, nodesize, maxnodes, k_folds = 5) {
  folds <- createFolds(data[[target]], k = k_folds, list = TRUE)
  cv_scores <- numeric(k_folds)
  
  for (i in seq_along(folds)) {
    train_idx <- unlist(folds[-i])
    valid_idx <- folds[[i]]
    
    train_fold <- data[train_idx, ]
    valid_fold <- data[valid_idx, ]
    
    # Configurar parámetros del Random Forest
    rf_params <- list(
      x = train_fold %>% select(-all_of(target)),
      y = train_fold[[target]],
      ntree = ntree,
      mtry = mtry,
      sampsize = round(nrow(train_fold) * sampsize),
      nodesize = nodesize,
      importance = TRUE
    )
    
    # Agregar maxnodes solo si no es NA
    if (!is.na(maxnodes)) {
      rf_params$maxnodes <- maxnodes
    }
    
    
    # Entrenar Random Forest (Randomforest aplica Gini por default)
        # La librería ejecuta los pasos de bootstrap + selección aleatoria de variables + crecimiento sin poda.
    rf_model <- do.call(randomForest, rf_params)
    
    # Predecir en validación 
      #Agregación de los árboles: aplicando la moda de las predicciones individuales de los árboles en el predict().
    pred <- predict(rf_model, newdata = valid_fold %>% select(-all_of(target)))
    
    # Calcular accuracy
    cv_scores[i] <- mean(pred == valid_fold[[target]])
  }
  
  return(mean(cv_scores))
}

# 8.3 Grid Search con validación cruzada
# El Grid Search evalúa sistemáticamente todas las combinaciones de
# hiperparámetros para encontrar la configuración óptima. Es computacionalmente
# costoso pero garantiza encontrar la mejor combinación dentro del espacio definido.

# Inicializar vector para almacenar resultados
cv_results <- numeric(nrow(grid_rf))

# Ejecutar Grid Search 
      #Ajuste de hiperparámetros (tuning) mediante validación cruzada
for (i in 1:nrow(grid_rf)) { 
  cat("Probando combinación", i, "de", nrow(grid_rf), "\n")
  
  cv_results[i] <- train_rf_cv(
    data = datos_rf,
    target = target_rf,
    mtry = grid_rf$mtry[i],
    ntree = grid_rf$ntree[i],
    sampsize = grid_rf$sampsize[i],
    nodesize = grid_rf$nodesize[i],
    maxnodes = grid_rf$maxnodes[i],
    k_folds = 5
  )
  
  cat("CV Score:", round(cv_results[i], 4), "\n")
}

# 8.4 Selección de mejores hiperparámetros (Resultado del grid search: mejor combinación)
# Se selecciona la combinación de hiperparámetros que maximiza el
# rendimiento en validación cruzada, proporcionando la mejor generalización.

grid_rf$cv_score <- cv_results
best_idx <- which.max(cv_results)
print(grid_rf[best_idx, ])
# Mejor CV Score
print(round(max(cv_results), 4))

# Ordenar por CV Score para ver top 5
top_5 <- grid_rf[order(-grid_rf$cv_score), 1:6]
print(top_5)

# 8.5 Entrenamiento del modelo final con mejores hiperparámetros
# Una vez identificados los hiperparámetros óptimos, se entrena el modelo
# final con toda la información disponible para maximizar el rendimiento.


# Configurar parámetros óptimos
best_params <- grid_rf[best_idx, ]

rf_final <- randomForest(
  x = datos_rf %>% select(-all_of(target_rf)),
  y = datos_rf[[target_rf]],
  ntree = best_params$ntree,
  mtry = best_params$mtry,
  sampsize = round(nrow(datos_rf) * best_params$sampsize),
  nodesize = best_params$nodesize,
  maxnodes = if (is.na(best_params$maxnodes)) NULL else best_params$maxnodes,
  importance = TRUE,
  proximity = TRUE
)

# 8.6 Análisis del modelo final
# El análisis del modelo final incluye evaluación de estabilidad,
# importancia de variables y rendimiento en datos no vistos.


# Información básica del modelo
  # Número de árboles
  print(rf_final$ntree)
  #Numero de variables por división(mtry)
  print(rf_final$mtry)
  # Errpr OOB:
  print(round(rf_final$err.rate[rf_final$ntree, "OOB"], 4)) # Estimación del error Out-of-Bag (OOB)

  # El error OOB es una estimación no sesgada del error de generalización
  # que se calcula usando las observaciones que no fueron incluidas en el bootstrap
  # de cada árbol individual.
  

# Matriz de confusión OOB
print(rf_final$confusion)

# Importancia de variables (Mean Decrease Gini)
# La importancia de variables en Random Forest se basa en la reducción
# promedio de impureza (Gini) que cada variable contribuye a través de todos los árboles.

importance_df <- data.frame(
  variable = rownames(importance(rf_final)),
  importance = importance(rf_final)[, "MeanDecreaseGini"]
) %>% arrange(desc(importance))

print(head(importance_df, 10))

# Visualización de importancia
ggplot(head(importance_df, 15), aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Variables más Importantes",
       x = "Variable",
       y = "Importancia (Mean Decrease Gini)") +
  theme_minimal()

# 8.7 Evaluación en conjunto de test
# La evaluación en datos de prueba proporciona una estimación realista
# del rendimiento del modelo en producción, incluyendo análisis de umbrales óptimos.


test_rf <- test_data %>% dplyr::select(-cliente_id)

# Probabilidad en test
pred_rf_prob <- predict(rf_final, newdata = test_rf %>% select(-all_of(target_rf)), type = "prob")[, "Si"]

# Análisis ROC y AUC
# La curva ROC muestra la relación entre sensibilidad (TPR) y especificidad (1-FPR)
# para diferentes umbrales de clasificación. El AUC mide la capacidad discriminativa del modelo.

roc_obj_rf <- pROC::roc(response = test_rf[[target_rf]], predictor = pred_rf_prob, levels = c("No", "Si"))
auc_val <- as.numeric(pROC::auc(roc_obj_rf))
cat("AUC =", round(auc_val, 4), "\n")

# Visualización de curva ROC
plot(roc_obj_rf, main = "Curva ROC - Random Forest", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

#8.8 Optimización de umbral de clasificación
# El umbral óptimo se selecciona según el criterio de Youden, que maximiza
# la suma de sensibilidad y especificidad menos 1. Esto es especialmente importante
# en problemas de churn donde el costo de falsos negativos puede ser alto.

# Encontrar threshold óptimo según Youden (para mejorar la sensibilidad - detectar más fugados, sacrificando sensibilidad)
thr_best <- as.numeric(pROC::coords(roc_obj_rf, "best", best.method = "youden")["threshold"])
cat("Threshold Youden =", round(thr_best, 4), "\n")

# Aplicar ese threshold
pred_rf_class <- factor(ifelse(pred_rf_prob > thr_best, "Si", "No"),
                        levels = c("No", "Si"))

# Predicciones
pred_05   <- factor(ifelse(pred_rf_prob > 0.5,     "Si", "No"), levels = c("No", "Si"))
pred_best <- factor(ifelse(pred_rf_prob > thr_best, "Si", "No"), levels = c("No", "Si"))

# Matriz de confusión
cm_05   <- caret::confusionMatrix(pred_05,   test_rf[[target_rf]], positive = "Si")
print(cm_05)
cm_best <- caret::confusionMatrix(pred_best, test_rf[[target_rf]], positive = "Si")
print(cm_best)

# 8.9 Métricas adicionales
# Las métricas F1 proporcionan un balance entre precisión y sensibilidad,
# especialmente importante en problemas de clasificación desbalanceada como churn.

    ## threshold = 0.5
f1_si_rf_05 = 2 * (cm_05$byClass["Pos Pred Value"] * cm_05$byClass["Sensitivity"]) /
  (cm_05$byClass["Pos Pred Value"] + cm_05$byClass["Sensitivity"])
f1_no_rf_05 = 2 * (cm_05$byClass["Neg Pred Value"] * cm_05$byClass["Specificity"]) /
  (cm_05$byClass["Neg Pred Value"] + cm_05$byClass["Specificity"])
    ## threshold = best
f1_si_rf_best = 2 * (cm_best$byClass["Pos Pred Value"] * cm_best$byClass["Sensitivity"]) /
  (cm_best$byClass["Pos Pred Value"] + cm_best$byClass["Sensitivity"])
f1_no_rf_best = 2 * (cm_best$byClass["Neg Pred Value"] * cm_best$byClass["Specificity"]) /
  (cm_best$byClass["Neg Pred Value"] + cm_best$byClass["Specificity"])


# 8.10 Guardo métricas en el df para comparar modelos
# La consolidación de métricas permite comparar diferentes configuraciones
# del modelo y seleccionar la mejor para producción.

# Actualizo tabla de métricas
  # Con threshold=0.5
metricas_modelos <- rbind(metricas_modelos, data.frame(
  modelo = "Random Forest (Grid Search + CV) - threshold=0.5",
  accuracy = cm_05$overall["Accuracy"],
  error_rate = 1 - conf_matrix$overall["Accuracy"],
  kappa = cm_05$overall["Kappa"],
  sensibilidad = cm_05$byClass["Sensitivity"],
  especificidad = cm_05$byClass["Specificity"],
  precision_si = as.numeric(cm_05$byClass["Pos Pred Value"]),
  f1_no = f1_no_rf_05,
  f1_si = f1_si_rf_05,
  balanced_accuracy = cm_05$byClass["Balanced Accuracy"],
  auc = auc_val,
  row.names = NULL
))

# Con threshold=óptimo
metricas_modelos <- rbind(metricas_modelos, data.frame(
  modelo = "Random Forest (Grid Search + CV) - threshold=óptimo",
  accuracy = cm_best$overall["Accuracy"],
  error_rate = 1 - cm_best$overall["Accuracy"],
  kappa = cm_best$overall["Kappa"],
  sensibilidad = cm_best$byClass["Sensitivity"],
  especificidad = cm_best$byClass["Specificity"],
  precision_si = as.numeric(cm_best$byClass["Pos Pred Value"]),
  f1_no = f1_no_rf_best,
  f1_si = f1_si_rf_best,
  balanced_accuracy = cm_best$byClass["Balanced Accuracy"],
  auc = auc_val,
  row.names = NULL
))


print(metricas_modelos)


# 8.11 Guardado del modelo 
# La persistencia del modelo incluye no solo el objeto entrenado, sino también
# metadatos necesarios para reproducibilidad, interpretación y despliegue


rf_artifact <- list(
  model_type      = "randomForest",
  model           = rf_final,
  # resultados de tuning
  grid_results    = grid_rf,
  best_idx        = best_idx,
  best_params     = best_params,
  var_importance  = importance_df,
  # esquema de datos
  feature_names   = colnames(datos_rf %>% dplyr::select(-all_of(target_rf))),
  target_name     = target_rf,
  target_levels   = levels(train_data[[target_rf]]),
  # umbral elegido y métricas en test (dos escenarios)
  eval = list(
    auc            = as.numeric(auc_val),
    thr_best       = as.numeric(thr_best),
    cm_thr05       = cm_05,
    cm_thr_best    = cm_best
  ),
  # housekeeping
  seed            = 123,
  session         = sessionInfo(),
  created_at      = Sys.time()
)

saveRDS(rf_artifact, here("artifacts", "rf_bundle_v1.rds"), compress = "xz")



# 9. Desarrollo de modelo: REGRESIÓN LOGÍSTICA ------------------------------
#
# La regresión logística es un modelo lineal generalizado que modela
# la probabilidad de pertenecer a una clase usando la función logística:
# - Función logística: P(Y=1) = 1/(1 + e^(-z)) donde z = β₀ + β₁X₁ + ... + βₖXₖ
# - Odds ratio: e^βᵢ representa cuántas veces aumenta la probabilidad de Y=1 por unidad de Xᵢ
# - Ventajas: Interpretabilidad, probabilidades calibradas, eficiencia computacional
# - Desventajas: Asume relación lineal, sensible a outliers, requiere selección de variables


set.seed(123)

# 9.1 Preparo de datos para regresión logística
datos_lr <- train_data_bal %>% dplyr::select(-cliente_id)
target_lr <- "fugado"

# Convierto variables categóricas a dummy variables (one-hot encoding)
# Para regresión logística necesitamos variables numéricas

  # Creo dummy variables solo para factores categóricos
  datos_lr_encoded <- model.matrix(~ provincia + tipo_cuenta + rubro - 1, data = datos_lr) %>%  #El `-1` elimina la categoría de referencia para evitar **multicolinealidad perfecta**
  as.data.frame() %>%
  
  # Agrego el resto de variables (numéricas y lógicas)
  bind_cols(
    datos_lr %>% select(-provincia, -tipo_cuenta, -rubro) %>%
      mutate(across(c(ventas_online, ventas_presencial, ventas_mixtas, usuario_nuevo), as.numeric))
  ) %>%
  
  # Agrego la variable target
  mutate(fugado = datos_lr$fugado)


  # 9.2 Análisis de colinealidad (VIF - Factor de Inflación de Varianza)
        #Tengo variables altamente correlacionadas que crean dependencias lineales casi perfectas. 
        #aunque la matriz tiene rango completo, las correlaciones muy altas (>0.9) causan inestabilidad numérica en el cálculo del VIF.
        #VIF (Variance Inflation Factor): Mide cuánto se infla la varianza de un coeficiente debido a la colinealidad
        #Regla práctica: VIF > 5 indica colinealidad problemática, VIF > 10 indica colinealidad severa
        #Si X₁ y X₂ están altamente correlacionadas, es difícil distinguir su efecto individual en Y
  
  if (requireNamespace("car", quietly = TRUE)) {
    # Verifico estructura de los datos
    cat("Estructura de datos_lr_encoded:\n")
    str(datos_lr_encoded)
    cat("\nNombres de variables:\n")
    print(names(datos_lr_encoded))
    
    # Creo una versión sin variables altamente correlacionadas para VIF
    datos_vif <- datos_lr_encoded %>% 
      select(-fugado) %>%
      select_if(function(x) var(x, na.rm = TRUE) > 0) %>%
      select(-matches("provincia[^_]*$")) %>%
      select(-matches("tipo_cuenta[^_]*$")) %>%
      select(-matches("rubro[^_]*$")) %>%
      mutate(fugado = datos_lr_encoded$fugado)
    
    # Identifico y elimino variables altamente correlacionadas (umbral 0.9)
    cor_matrix <- cor(datos_vif %>% select(-fugado), use = "complete.obs")
    
    # Función para encontrar variables altamente correlacionadas
    encontrar_correlaciones_altas <- function(cor_matrix, umbral = 0.9) {
      pares_cor_altos <- which(abs(cor_matrix) > umbral & cor_matrix != 1, arr.ind = TRUE)
      
      if (nrow(pares_cor_altos) == 0) {
        return(character(0))
      }
      
      pares_cor <- data.frame(
        var1 = rownames(cor_matrix)[pares_cor_altos[,1]],
        var2 = colnames(cor_matrix)[pares_cor_altos[,2]],
        cor_val = cor_matrix[pares_cor_altos]
      ) %>%
        mutate(abs_cor = abs(cor_val)) %>%
        arrange(desc(abs_cor))
      
      vars_a_eliminar <- character(0)
      for (i in 1:nrow(pares_cor)) {
        var1 <- pares_cor$var1[i]
        var2 <- pares_cor$var2[i]
        if (!var1 %in% vars_a_eliminar && !var2 %in% vars_a_eliminar) {
          vars_a_eliminar <- c(vars_a_eliminar, var2)
        }
      }
      
      return(unique(vars_a_eliminar))
    }
    
    # Identifico variables altamente correlacionadas (umbral 0.9)
    vars_a_eliminar <- encontrar_correlaciones_altas(cor_matrix, umbral = 0.9)
    
    cat("\nVariables altamente correlacionadas (>0.9) que se eliminarán:\n")
    print(vars_a_eliminar)
    
    # Creo dataset final para VIF sin variables altamente correlacionadas
    datos_vif_final <- datos_vif %>%
      select(-all_of(vars_a_eliminar))
    
    cat("\nVariables finales para análisis VIF:\n")
    print(names(datos_vif_final %>% select(-fugado)))
    
    # Verifico que no hay problemas de rango
    X_matrix <- as.matrix(datos_vif_final %>% select(-fugado))
    rank_X <- qr(X_matrix)$rank
    n_vars <- ncol(X_matrix)
    
    cat("\nRango de la matriz de diseño:", rank_X, "de", n_vars, "variables\n")
    
    # Calculo VIF
    if (rank_X == n_vars && n_vars > 0) {
      modelo_vif <- glm(fugado ~ . - fugado, 
                        data = datos_vif_final, 
                        family = binomial(link = "logit"))
      
      vif_scores <- car::vif(modelo_vif)
      vif_df <- data.frame(
        variable = names(vif_scores),
        vif = as.numeric(vif_scores)
      ) %>% arrange(desc(vif))
      
      print("Top 10 variables con mayor VIF:")
      print(head(vif_df, 10))
      
      # Identifico variables problemáticas (VIF > 5)
      vars_problematicas <- vif_df$variable[vif_df$vif > 5]
      cat("Variables con VIF > 5:", length(vars_problematicas), "\n")
      if (length(vars_problematicas) > 0) {
        print(vars_problematicas)
      }
      
    } else {
      cat("No se puede calcular VIF debido a problemas de rango en la matriz de diseño.\n")
    }
  }
  
  # 9.3 Selección de variables con LASSO (Regularización L1)
  # Con muchas variables, el modelo puede sobreajustarse (overfitting).
  # LASSO combina regularización L1 con selección automática de variables:
  # - Penalización L1: min(β) [log-likelihood + λ∑|βⱼ|]
  # - Algunos coeficientes se reducen exactamente a 0, eliminando variables automáticamente
  # - Ventaja: Selección automática + prevención de overfitting en un solo paso
  
  
  if (requireNamespace("glmnet", quietly = TRUE)) {
    # Preparo datos
    X_lasso_vif <- as.matrix(datos_vif_final %>% select(-fugado))
    y_lasso <- as.numeric(datos_vif_final$fugado == "Si")
    
    cat("\n=== SELECCIÓN CON LASSO ===\n")
    cat("Variables para LASSO:", ncol(X_lasso_vif), "\n")
    
    # Entreno modelo LASSO
    set.seed(123)
    cv_lasso <- cv.glmnet(X_lasso_vif, y_lasso, 
                          family = "binomial", 
                          alpha = 1,  # LASSO
                          nfolds = 5,
                          type.measure = "auc")
    
    # Lambda óptimo
    lambda_opt <- cv_lasso$lambda.1se
    cat("Lambda óptimo (1SE):", round(lambda_opt, 6), "\n")
    
    # Coeficientes del modelo LASSO 
    coef_lasso <- coef(cv_lasso, s = lambda_opt)
    coef_df <- data.frame(
      variable = rownames(coef_lasso),
      coeficiente = as.numeric(coef_lasso)
    ) %>%
      filter(coeficiente != 0) %>%
      arrange(desc(abs(coeficiente)))
    
    print("Variables seleccionadas por LASSO:")
    print(coef_df)
    
    # Variables seleccionadas (excluyendo intercept)
    vars_seleccionadas <- coef_df$variable[coef_df$variable != "(Intercept)" & coef_df$coeficiente != 0]
    cat("Número de variables seleccionadas:", length(vars_seleccionadas), "\n")
    
    # Creo dataset final con variables seleccionadas por LASSO
    datos_lr_final <- datos_vif_final %>%
      select(all_of(vars_seleccionadas), fugado)
    
  } else {
    cat("glmnet no disponible, usando variables pre-filtradas por VIF\n")
    datos_lr_final <- datos_vif_final
  }
  
  
  # 9.4 Refinamiento del modelo final - Eliminar variables con VIF alto
  # Este paso adicional es necesario porque LASSO puede mantener variables
  # colineales si individualmente son predictivas. LASSO no elimina colinealidad,
  # solo selecciona variables. La eliminación manual de variables con VIF > 5
  # mejora la estabilidad numérica del modelo.
  
  
  # Variables con VIF problemático que LASSO mantuvo
  vars_vif_alto <- c("ventas_en_fin_de_semana", "monto_total_historico", "ticket_promedio")
  
  # Creo modelo final refinado
  datos_lr_final_refinado <- datos_lr_final %>%
    select(-all_of(vars_vif_alto))
  
  cat("Variables eliminadas por VIF alto:", length(vars_vif_alto), "\n")
  cat("Variables finales:", ncol(datos_lr_final_refinado) - 1, "\n")
  print(names(datos_lr_final_refinado %>% select(-fugado)))
  
  
  # Verifico VIF del modelo final
  if (requireNamespace("car", quietly = TRUE)) {
    modelo_final <- glm(fugado ~ . - fugado, 
                        data = datos_lr_final, 
                        family = binomial(link = "logit"))
    
    vif_final <- car::vif(modelo_final)
    vif_final_df <- data.frame(
      variable = names(vif_final),
      vif = as.numeric(vif_final)
    ) %>% arrange(desc(vif))
    
    print("VIF del modelo final (LASSO):")
    print(vif_final_df)
    
    # Verifico si aún hay variables problemáticas
    vars_problematicas_refinado <- vif_final_df$variable[vif_final_df$vif > 5]
    cat("Variables con VIF > 5 en modelo refinado:", length(vars_problematicas_refinado), "\n")
    if (length(vars_problematicas_refinado) > 0) {
      print(vars_problematicas_refinado)
      cat("Aún hay variables problemáticas. Considera eliminar más.\n")
    } else {
      cat("Modelo refinado: Todas las variables tienen VIF < 5\n")
    }
  }
  
  # Actualizo el dataset final
  datos_lr_final <- datos_lr_final_refinado
  
  # Resumen final del proceso
  cat("\n=== RESUMEN DEL PROCESO DE SELECCIÓN ===\n")
  cat("1. Variables originales:", ncol(datos_lr_encoded) - 1, "\n")
  cat("2. Después de filtro VIF (0.9):", ncol(datos_vif_final) - 1, "\n")
  cat("3. Seleccionadas por LASSO:", ncol(datos_lr_final) + length(vars_vif_alto) - 1, "\n")
  cat("4. Eliminadas por VIF alto:", length(vars_vif_alto), "\n")
  cat("5. Variables finales en el modelo:", ncol(datos_lr_final) - 1, "\n")
  
  cat("\nProceso de selección de variables completado\n")
  cat("Modelo listo para entrenamiento final\n")
 
# 9.5 Entrenamiento del modelo de regresión logística
      # Función logística: `P(Y=1) = 1/(1 + e^(-z))` donde `z = β₀ + β₁X₁ + ... + βₖXₖ`
      # Odds ratio: `e^βᵢ` representa cuántas veces aumenta la probabilidad de Y=1 por unidad de Xᵢ
      # Interpretación: βᵢ > 0: Variable aumenta probabilidad de fuga; βᵢ < 0: Variable disminuye probabilidad de fuga
  
  
set.seed(123)

# Modelo con variables seleccionadas
modelo_lr <- glm(fugado ~ ., 
                 data = datos_lr_final, 
                 family = binomial(link = "logit"))

# Resumen del modelo
print(summary(modelo_lr))

# 9.6 Análisis de significancia de variables
# El test de Wald evalúa la significancia estadística de cada coeficiente:
  # ¿Por qué p-valores? Test de Wald: H₀: βᵢ = 0 vs H₁: βᵢ ≠ 0
  # p < 0.05: Rechazamos H₀, la variable es estadísticamente significativa
  # Interpretación: Si p < 0.05, la variable tiene efecto real en la probabilidad de fuga


# Extraer p-valores
coef_summary <- summary(modelo_lr)$coefficients
coef_df <- data.frame(
  variable = rownames(coef_summary),
  coeficiente = coef_summary[, "Estimate"],
  p_valor = coef_summary[, "Pr(>|z|)"],
  significativo = coef_summary[, "Pr(>|z|)"] < 0.05
) %>%
  arrange(p_valor)

print("Variables ordenadas por significancia:")
print(coef_df)

# Variables significativas
vars_significativas <- coef_df$variable[coef_df$significativo & coef_df$variable != "(Intercept)"]
cat("Variables significativas (p < 0.05):", length(vars_significativas), "\n")

# 9.7 Evaluación del modelo en entrenamiento
# La evaluación en entrenamiento proporciona una línea base del rendimiento
# del modelo, aunque puede estar optimista debido a overfitting.


# Predicciones en entrenamiento
pred_lr_train_prob <- predict(modelo_lr, type = "response")
pred_lr_train_class <- factor(ifelse(pred_lr_train_prob > 0.5, "Si", "No"),
                             levels = c("No", "Si"))

# Matriz de confusión en entrenamiento
cm_train <- confusionMatrix(pred_lr_train_class, datos_lr_final$fugado, positive = "Si")
print("Matriz de confusión en entrenamiento:")
print(cm_train)

# 9.8 Evaluación en conjunto de test
# La evaluación en datos de prueba proporciona una estimación realista
# del rendimiento del modelo en producción.


# Preparar datos de test con las mismas variables
test_lr <- test_data %>% dplyr::select(-cliente_id)


test_lr_encoded <- test_lr %>%
  mutate(
    ventas_online = as.numeric(ventas_online),
    ventas_presencial = as.numeric(ventas_presencial), 
    ventas_mixtas = as.numeric(ventas_mixtas),
    usuario_nuevo = as.numeric(usuario_nuevo)
  ) %>%
 
  model.matrix(~ provincia + tipo_cuenta + rubro - 1, data = .) %>%
  as.data.frame() %>%
  # Agrego el resto de variables (numéricas y lógicas)
  bind_cols(
    test_lr %>% select(-provincia, -tipo_cuenta, -rubro) %>%
      mutate(across(c(ventas_online, ventas_presencial, ventas_mixtas, usuario_nuevo), as.numeric))
  ) %>%
  # Agrego la variable target
  mutate(fugado = test_lr$fugado)

# Selecciono solo las variables del modelo final
vars_modelo_final <- names(datos_lr_final %>% select(-fugado))
test_lr_final <- test_lr_encoded %>%
  select(all_of(vars_modelo_final), fugado)

# Verifico que las variables coincidan
cat("Variables en modelo:", length(vars_modelo_final), "\n")
cat("Variables en test:", ncol(test_lr_final) - 1, "\n")
cat("Variables coinciden:", all(vars_modelo_final %in% names(test_lr_final %>% select(-fugado))), "\n")

# Predicciones en test
pred_lr_test_prob <- predict(modelo_lr, newdata = test_lr_final, type = "response")
pred_lr_test_class <- factor(ifelse(pred_lr_test_prob > 0.5, "Si", "No"),
                             levels = c("No", "Si"))

# Matriz de confusión en test
cm_test <- confusionMatrix(pred_lr_test_class, test_lr_final$fugado, positive = "Si")
print("Matriz de confusión en test:")
print(cm_test)

# 9.9 Análisis de umbrales óptimos
# La optimización del umbral de clasificación es crucial para el rendimiento:
# - Threshold = 0.5: Equilibrio entre sensibilidad y especificidad
# - Threshold óptimo (Youden): Maximiza Sensibilidad + Especificidad - 1
# - En detección de churn, puede ser más importante detectar fugados (alta sensibilidad)
#   que evitar falsos positivos en términos de costo de negocio


# ROC y AUC
roc_obj_lr <- pROC::roc(response = test_lr_final$fugado, 
                        predictor = pred_lr_test_prob, 
                        levels = c("No", "Si"))
auc_lr <- as.numeric(pROC::auc(roc_obj_lr))
cat("AUC =", round(auc_lr, 4), "\n")

# Plot ROC
plot(roc_obj_lr, main = "Curva ROC - Regresión Logística", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

# Threshold óptimo según Youden
thr_opt_lr <- as.numeric(pROC::coords(roc_obj_lr, "best", best.method = "youden")["threshold"])
cat("Threshold óptimo (Youden):", round(thr_opt_lr, 4), "\n")

# Evaluar con diferentes umbrales
umbrales <- c(0.3, 0.4, 0.5, 0.6, thr_opt_lr)
resultados_umbrales <- data.frame()

for (thr in umbrales) {
  pred_thr <- factor(ifelse(pred_lr_test_prob > thr, "Si", "No"),
                    levels = c("No", "Si"))
  
  cm_thr <- confusionMatrix(pred_thr, test_lr_final$fugado, positive = "Si")
  
  f1_si <- 2 * (cm_thr$byClass["Pos Pred Value"] * cm_thr$byClass["Sensitivity"]) /
    (cm_thr$byClass["Pos Pred Value"] + cm_thr$byClass["Sensitivity"])
  
  resultados_umbrales <- rbind(resultados_umbrales, data.frame(
    umbral = thr,
    accuracy = as.numeric(cm_thr$overall["Accuracy"]),
    sensibilidad = as.numeric(cm_thr$byClass["Sensitivity"]),
    especificidad = as.numeric(cm_thr$byClass["Specificity"]),
    f1_si = as.numeric(f1_si)
  ))
}

print("Resultados por umbral:")
print(resultados_umbrales)

# 9.10 Predicciones con umbral óptimo
pred_lr_opt <- factor(ifelse(pred_lr_test_prob > thr_opt_lr, "Si", "No"),
                     levels = c("No", "Si"))

cm_opt <- confusionMatrix(pred_lr_opt, test_lr_final$fugado, positive = "Si")
print("Matriz de confusión con umbral óptimo:")
print(cm_opt)

# 9.11 Métricas adicionales
# Las métricas F1 proporcionan un balance entre precisión y sensibilidad,
# especialmente importante en problemas de clasificación desbalanceada.

# Métricas para threshold = 0.5
f1_si_lr_05 <- 2 * (cm_test$byClass["Pos Pred Value"] * cm_test$byClass["Sensitivity"]) /
  (cm_test$byClass["Pos Pred Value"] + cm_test$byClass["Sensitivity"])

f1_no_lr_05 <- 2 * (cm_test$byClass["Neg Pred Value"] * cm_test$byClass["Specificity"]) /
  (cm_test$byClass["Neg Pred Value"] + cm_test$byClass["Specificity"])

# Métricas para threshold óptimo
f1_si_lr_opt <- 2 * (cm_opt$byClass["Pos Pred Value"] * cm_opt$byClass["Sensitivity"]) /
  (cm_opt$byClass["Pos Pred Value"] + cm_opt$byClass["Sensitivity"])

f1_no_lr_opt <- 2 * (cm_opt$byClass["Neg Pred Value"] * cm_opt$byClass["Specificity"]) /
  (cm_opt$byClass["Neg Pred Value"] + cm_opt$byClass["Specificity"])


# 9.12 Visualización de importancia de variables e interpretación del modelo
# La interpretación de coeficientes en regresión logística es directa:
# - Coeficiente positivo: Variable aumenta probabilidad de churn
# - Coeficiente negativo: Variable disminuye probabilidad de churn
# - Odds Ratio: e^coeficiente representa el factor de cambio en la probabilidad


# Crear gráfico de coeficientes
coef_plot <- coef_df %>%
  filter(variable != "(Intercept)") %>%
  mutate(
    variable = reorder(variable, abs(coeficiente)),
    signo = ifelse(coeficiente > 0, "Factor de Riesgo", "Factor Protector")
  ) %>%
  ggplot(aes(x = variable, y = coeficiente, fill = signo)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Factor de Riesgo" = "#e74c3c", "Factor Protector" = "#27ae60")) +
  labs(
    title = "Importancia de Variables - Regresión Logística",
    subtitle = "Coeficientes estandarizados (más alto = más importante)",
    x = "Variables",
    y = "Coeficiente",
    fill = "Tipo de Factor"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.text.y = element_text(size = 10),
    legend.position = "bottom"
  )

print(coef_plot)

# Interpretación del modelo de Regresión Logística
# La interpretación de regresión logística incluye análisis de significancia,
# magnitud de efectos, y factores de riesgo vs protectores.
# Interpretación del modelo de Regresión Logística

cat("=== INTERPRETACIÓN DEL MODELO DE REGRESIÓN LOGÍSTICA ===\n")

#  a. Resumen del modelo
cat("\n1. RESUMEN DEL MODELO:\n")
print(summary(modelo_lr))

# b. Coeficientes y significancia
cat("\n2. COEFICIENTES Y SIGNIFICANCIA:\n")
coef_summary <- summary(modelo_lr)$coefficients
coef_interpretacion <- data.frame(
  variable = rownames(coef_summary),
  coeficiente = coef_summary[, "Estimate"],
  error_estandar = coef_summary[, "Std. Error"],
  z_valor = coef_summary[, "z value"],
  p_valor = coef_summary[, "Pr(>|z|)"],
  significativo = coef_summary[, "Pr(>|z|)"] < 0.05,
  odds_ratio = exp(coef_summary[, "Estimate"]),
  stringsAsFactors = FALSE
) %>%
  arrange(desc(abs(coeficiente)))

print(coef_interpretacion)

# c. Variables más importantes (por coeficiente absoluto)
cat("\n3. TOP 10 VARIABLES MÁS IMPORTANTES:\n")
top_variables <- coef_interpretacion %>%
  filter(variable != "(Intercept)") %>%
  head(10) %>%
  mutate(
    tipo_efecto = ifelse(coeficiente > 0, "Factor de Riesgo", "Factor Protector"),
    magnitud = case_when(
      abs(coeficiente) > 1 ~ "Alto",
      abs(coeficiente) > 0.5 ~ "Medio",
      TRUE ~ "Bajo"
    )
  )

print(top_variables)

# d. Interpretación de coeficientes
cat("\n4. INTERPRETACIÓN DE COEFICIENTES:\n")
cat("Los coeficientes representan el cambio en el log-odds por unidad de cambio en la variable.\n")
cat("Odds Ratio = e^coeficiente representa cuántas veces aumenta/disminuye la probabilidad de fuga.\n\n")

for (i in 1:min(5, nrow(top_variables))) {
  var <- top_variables$variable[i]
  coef <- top_variables$coeficiente[i]
  or <- top_variables$odds_ratio[i]
  sig <- top_variables$significativo[i]
  tipo <- top_variables$tipo_efecto[i]
  
  cat("•", var, ":\n")
  cat("  - Coeficiente:", round(coef, 4), "\n")
  cat("  - Odds Ratio:", round(or, 4), "\n")
  cat("  - Efecto:", tipo, "\n")
  cat("  - Significativo:", ifelse(sig, "SÍ", "NO"), "\n")
  
  if (coef > 0) {
    cat("  - Interpretación: Por cada unidad de aumento en", var, 
        ", la probabilidad de fuga aumenta", round((or-1)*100, 1), "%\n")
  } else {
    cat("  - Interpretación: Por cada unidad de aumento en", var, 
        ", la probabilidad de fuga disminuye", round((1-or)*100, 1), "%\n")
  }
  cat("\n")
}

# e. Variables significativas vs no significativas
cat("5. ANÁLISIS DE SIGNIFICANCIA:\n")
vars_significativas <- coef_interpretacion %>% filter(significativo & variable != "(Intercept)")
vars_no_significativas <- coef_interpretacion %>% filter(!significativo & variable != "(Intercept)")

cat("Variables significativas (p < 0.05):", nrow(vars_significativas), "\n")
cat("Variables no significativas (p >= 0.05):", nrow(vars_no_significativas), "\n")

if (nrow(vars_significativas) > 0) {
  cat("\nVariables significativas ordenadas por importancia:\n")
  print(vars_significativas %>% select(variable, coeficiente, odds_ratio, p_valor))
}

# f. Análisis de factores de riesgo vs protectores
cat("\n6. FACTORES DE RIESGO VS PROTECTORES:\n")
factores_riesgo <- top_variables %>% filter(tipo_efecto == "Factor de Riesgo")
factores_protectores <- top_variables %>% filter(tipo_efecto == "Factor Protector")

cat("Factores de Riesgo (aumentan probabilidad de fuga):\n")
if (nrow(factores_riesgo) > 0) {
  for (i in 1:nrow(factores_riesgo)) {
    cat("-", factores_riesgo$variable[i], "(OR =", round(factores_riesgo$odds_ratio[i], 3), ")\n")
  }
} else {
  cat("No hay factores de riesgo significativos en el top 10\n")
}

cat("\nFactores Protectores (disminuyen probabilidad de fuga):\n")
if (nrow(factores_protectores) > 0) {
  for (i in 1:nrow(factores_protectores)) {
    cat("-", factores_protectores$variable[i], "(OR =", round(factores_protectores$odds_ratio[i], 3), ")\n")
  }
} else {
  cat("No hay factores protectores significativos en el top 10\n")
}



# 9.13 Actualizar tabla de métricas
# La consolidación de métricas permite comparar diferentes configuraciones
# del modelo y seleccionar la mejor para producción.

metricas_modelos <- rbind(metricas_modelos, data.frame(
  modelo = "Regresión Logística (LASSO) - threshold=0.5",
  accuracy = cm_test$overall["Accuracy"],
  error_rate = 1 - cm_test$overall["Accuracy"],
  kappa = cm_test$overall["Kappa"],
  sensibilidad = cm_test$byClass["Sensitivity"],
  especificidad = cm_test$byClass["Specificity"],
  precision_si = as.numeric(cm_test$byClass["Pos Pred Value"]),
  f1_no = f1_no_lr_05,
  f1_si = f1_si_lr_05,
  balanced_accuracy = cm_test$byClass["Balanced Accuracy"],
  auc = auc_lr,
  row.names = NULL
))

metricas_modelos <- rbind(metricas_modelos, data.frame(
  modelo = "Regresión Logística (LASSO) - threshold=óptimo",
  accuracy = cm_opt$overall["Accuracy"],
  error_rate = 1 - cm_opt$overall["Accuracy"],
  kappa = cm_opt$overall["Kappa"],
  sensibilidad = cm_opt$byClass["Sensitivity"],
  especificidad = cm_opt$byClass["Specificity"],
  precision_si = as.numeric(cm_opt$byClass["Pos Pred Value"]),
  f1_no = f1_no_lr_opt,
  f1_si = f1_si_lr_opt,
  balanced_accuracy = cm_opt$byClass["Balanced Accuracy"],
  auc = auc_lr,
  row.names = NULL
))

print("Comparación de modelos:")
print(metricas_modelos)

# 9.14 Guardado del modelo
# La persistencia del modelo incluye no solo el objeto entrenado, sino también
# metadatos necesarios para reproducibilidad, interpretación y despliegue.


lr_artifact <- list(
  model_type = "glm",
  model = modelo_lr,
  # variables seleccionadas
  selected_vars = if(exists("vars_seleccionadas")) vars_seleccionadas else colnames(datos_lr_final)[-ncol(datos_lr_final)],
  # análisis de colinealidad
  vif_scores = if(exists("vif_df")) vif_df else NULL,
  # coeficientes y significancia
  coef_summary = coef_df,
  # umbral óptimo
  optimal_threshold = thr_opt_lr,
  # esquema de datos
  feature_names = colnames(datos_lr_final %>% select(-fugado)),
  target_name = "fugado",
  target_levels = levels(train_data$fugado),
  # métricas en test
  eval = list(
    auc = auc_lr,
    cm_thr05 = cm_test,
    cm_thr_opt = cm_opt,
    threshold_analysis = resultados_umbrales
  ),
  # housekeeping
  seed = 123,
  session = sessionInfo(),
  created_at = Sys.time()
)

saveRDS(lr_artifact, here("artifacts", "lr_bundle_v1.rds"), compress = "xz")


# 10. Comparación de modelos ----------------------------------------------
# La comparación sistemática de modelos es fundamental para seleccionar
# el mejor algoritmo para producción. Se evalúan múltiples métricas para obtener
# una visión completa del rendimiento:
# - AUC: Capacidad discriminativa general del modelo
# - Accuracy: Proporción de predicciones correctas
# - Sensibilidad: Capacidad de detectar casos positivos (churn)
# - Especificidad: Capacidad de detectar casos negativos (no churn)
# - F1-Score: Balance entre precisión y sensibilidad


# 10.1 Tabla comparativa
# La tabla comparativa permite identificar rápidamente el modelo con
# mejor rendimiento en diferentes métricas. El ordenamiento por AUC prioriza
# la capacidad discriminativa general del modelo.

print(metricas_modelos %>% 
        select(modelo, auc, accuracy, sensibilidad, especificidad) %>%
        arrange(desc(auc)))

# 10.2 Gráfico de comparación
# La visualización gráfica facilita la comparación visual de múltiples
# métricas simultáneamente, permitiendo identificar patrones y trade-offs
# entre diferentes aspectos del rendimiento.

comparacion_plot <- metricas_modelos %>%
  select(modelo, auc, accuracy, sensibilidad, especificidad) %>%
  pivot_longer(cols = c(auc, accuracy, sensibilidad, especificidad), 
               names_to = "metrica", values_to = "valor") %>%
  ggplot(aes(x = modelo, y = valor, fill = metrica)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Comparación de Modelos", x = "Modelo", y = "Valor") +
  theme_minimal()

print(comparacion_plot)

# 10.3 Importancia de variables por modelo
# La comparación de importancia de variables entre modelos proporciona
# insights sobre la consistencia de los patrones identificados y la robustez
# de las características más relevantes para la predicción de churn.


# Crear dataframes de importancia
importancias <- list()

# Random Forest: Importancia basada en Mean Decrease Gini
# En Random Forest, la importancia se calcula como la reducción promedio
# de impureza (Gini) que cada variable contribuye a través de todos los árboles.
if (exists("importance_df") && !is.null(importance_df) && nrow(importance_df) > 0) {
  rf_importance <- importance_df %>%
    head(10) %>%
    mutate(modelo = "Random Forest")
  importancias[["rf"]] <- rf_importance
}

# Árbol de Decisión: Importancia basada en reducción de impureza
# En árboles de decisión, la importancia se basa en la reducción total
# de impureza que cada variable contribuye en todas las divisiones del árbol.
if (exists("tree_importance_df") && !is.null(tree_importance_df) && nrow(tree_importance_df) > 0) {
  tree_importance_plot <- tree_importance_df %>%
    head(10) %>%
    mutate(modelo = "Árbol de Decisión")
  importancias[["tree"]] <- tree_importance_plot
}

# Regresión Logística: Importancia basada en magnitud de coeficientes
# En regresión logística, la importancia se basa en la magnitud absoluta
# de los coeficientes. Se normaliza x1000 para visualización comparativa.

if (exists("coef_df") && !is.null(coef_df) && nrow(coef_df) > 0) {
  lr_importance <- coef_df %>%
    filter(variable != "(Intercept)") %>%
    head(10) %>%
    mutate(
      modelo = "Regresión Logística",
      importance = abs(coeficiente) * 1000  # ← NORMALIZACIÓN
    ) %>%
    select(variable, importance, modelo)
  importancias[["lr"]] <- lr_importance
}

# Combinar y visualizar
if (length(importancias) > 0) {
  importancia_combinada <- bind_rows(importancias)
  
  importancia_plot <- importancia_combinada %>%
    ggplot(aes(x = reorder(variable, importance), y = importance, fill = modelo)) +
    geom_col(position = "dodge") +
    coord_flip() +
    labs(title = "Top 10 Variables más Importantes por Modelo",
         subtitle = "RL normalizada x1000 para visualización",
         x = "Variable", y = "Importancia") +
    theme_minimal()
  
  print(importancia_plot)
}

# 10.4 Matrices de confusión de las 5 configuraciones
# Las matrices de confusión proporcionan información detallada sobre
# el rendimiento del modelo en términos de verdaderos/falsos positivos y negativos.
# Esta información es crucial para entender los trade-offs del modelo.

# Función para mostrar matriz de confusión
mostrar_matriz <- function(modelo_nombre, cm) {
  cat("\n---", modelo_nombre, "---\n")
  print(cm$table)
  cat("Accuracy:", round(cm$overall["Accuracy"], 4), "\n")
  cat("Sensibilidad:", round(cm$byClass["Sensitivity"], 4), "\n")
  cat("Especificidad:", round(cm$byClass["Specificity"], 4), "\n")
}

# Mostrar matrices de confusión
for (i in 1:nrow(metricas_modelos)) {
  modelo_nombre <- metricas_modelos$modelo[i]
  
  if (grepl("Random Forest.*threshold=0.5", modelo_nombre)) {
    mostrar_matriz(modelo_nombre, cm_05)
  } else if (grepl("Random Forest.*threshold=óptimo", modelo_nombre)) {
    mostrar_matriz(modelo_nombre, cm_best)
  } else if (grepl("Regresión Logística.*threshold=0.5", modelo_nombre)) {
    mostrar_matriz(modelo_nombre, cm_test)
  } else if (grepl("Regresión Logística.*threshold=óptimo", modelo_nombre)) {
    mostrar_matriz(modelo_nombre, cm_opt)
  } else if (grepl("Árbol", modelo_nombre)) {
    mostrar_matriz(modelo_nombre, conf_matrix)
  }
}

# 10.5 Mejor modelo
# La selección del mejor modelo se basa en el AUC como métrica principal,
# ya que proporciona una medida general de la capacidad discriminativa del modelo
# independientemente del umbral de clasificación elegido.

mejor_idx <- which.max(metricas_modelos$auc)
cat("\n=== MEJOR MODELO ===\n")
cat("Modelo:", metricas_modelos$modelo[mejor_idx], "\n")
cat("AUC:", metricas_modelos$auc[mejor_idx], "\n")
cat("Accuracy:", metricas_modelos$accuracy[mejor_idx], "\n")
cat("Sensibilidad:", metricas_modelos$sensibilidad[mejor_idx], "\n")
cat("Especificidad:", metricas_modelos$especificidad[mejor_idx], "\n")

# 10.6 Recomendación final
# La recomendación final considera no solo el rendimiento estadístico,
# sino también aspectos prácticos como interpretabilidad, robustez y facilidad
# de implementación en un entorno de producción.

cat("\n=== RECOMENDACIÓN FINAL ===\n")
cat("Usar", metricas_modelos$modelo[mejor_idx], "para producción\n")
cat("Justificación: Mayor AUC y balance entre sensibilidad y especificidad\n")

# 10.7 Guardado final de resultados
# La persistencia de resultados es crucial para reproducibilidad,
# auditoría y despliegue en producción. Se guardan tanto los modelos entrenados
# como las métricas y visualizaciones generadas.

# Crear directorio de resultados si no existe
dir.create(here("resultados"), showWarnings = FALSE)

# Guardar tabla de métricas
write.csv(metricas_modelos, here("resultados", "metricas_modelos.csv"), row.names = FALSE)
cat("✓ Métricas guardadas en: resultados/metricas_modelos.csv\n")

# Guardar gráficos
ggsave(here("resultados", "comparacion_modelos.png"), comparacion_plot, 
       width = 12, height = 8, dpi = 300)
cat("✓ Gráfico de comparación guardado en: resultados/comparacion_modelos.png\n")

if (exists("importancia_combinada")) {
  ggsave(here("resultados", "importancia_variables.png"), importancia_plot, 
         width = 12, height = 8, dpi = 300)
  cat("✓ Gráfico de importancia guardado en: resultados/importancia_variables.png\n")
}

# 10.8 Guardar resumen ejecutivo
# El resumen ejecutivo consolida los hallazgos principales del análisis
# en un formato accesible para stakeholders no técnicos, incluyendo recomendaciones
# de negocio y métricas clave.

resumen_ejecutivo <- list(
  fecha_analisis = Sys.time(),
  mejor_modelo = metricas_modelos$modelo[mejor_idx],
  mejor_auc = metricas_modelos$auc[mejor_idx],
  mejor_accuracy = metricas_modelos$accuracy[mejor_idx],
  mejor_sensibilidad = metricas_modelos$sensibilidad[mejor_idx],
  mejor_especificidad = metricas_modelos$especificidad[mejor_idx],
  total_modelos_evaluados = nrow(metricas_modelos),
  recomendacion = paste("Usar", metricas_modelos$modelo[mejor_idx], "para producción"),
  archivos_generados = c(
    "metricas_modelos.csv",
    "comparacion_modelos.png",
    "importancia_variables.png"
  )
)

saveRDS(resumen_ejecutivo, here("resultados", "resumen_ejecutivo.rds"))



# 11. Cálculos adicionales ----------------------------------------------------


## VALIDACION GINI VS ENTROPIA
# Funciones básicas
gini <- function(p) sum(p * (1 - p))
entropia <- function(p) -sum(p[p > 0] * log2(p[p > 0]))
entropia_norm <- function(p) {
  e <- entropia(p)
  max_e <- log2(length(p))
  (e / max_e) * 0.5  # Normalizar al rango [0, 0.5] como Gini
}

# Ejemplos de distribuciones
ejemplos <- list(
  "Puro (1,0)" = c(1, 0),
  "Sesgado (0.8,0.2)" = c(0.8, 0.2),
  "Uniforme (0.5,0.5)" = c(0.5, 0.5),
  "Tres clases" = c(0.6, 0.3, 0.1)
)

# Calcular y comparar
resultados <- data.frame()

for (nombre in names(ejemplos)) {
  p <- ejemplos[[nombre]]
  p <- p / sum(p)  # Normalizar
  
  g <- gini(p)
  e <- entropia(p)
  e_norm <- entropia_norm(p)
  
  resultados <- rbind(resultados, data.frame(
    distribucion = nombre,
    gini = round(g, 4),
    entropia = round(e, 4),
    entropia_norm = round(e_norm, 4),
    diferencia = round(abs(g - e_norm), 4)
  ))
}

print(resultados)

## IMPACTO EN NEGOCIO
promedios_por_cliente <- ventas %>%
  filter(estado == "APPROVED") %>%
  mutate(mes = floor_date(as.Date(fecha_venta), "month")) %>%
  group_by(cliente_id, mes) %>%
  summarise(
    tx_mes = n(),                          
    volumen_mes = sum(monto_venta, na.rm = TRUE), 
    .groups = "drop_last"
  ) %>%
  summarise(
    meses_activos = n(),
    total_tx = sum(tx_mes),
    total_volumen = sum(volumen_mes),
    avg_tx_mes = mean(tx_mes),                 
    avg_volumen_mes = mean(volumen_mes),      
    .groups = "drop"
  )


resumen_global <- promedios_por_cliente %>%
  summarise(
    transacciones_promedio = mean(avg_tx_mes, na.rm = TRUE),
    volumen_promedio_usd   = mean(avg_volumen_mes, na.rm = TRUE)
  )


## comparativa AUC-ROC
# Obtener probabilidades de cada modelo en test
pred_arbol_prob <- predict(arbol_final, newdata = test_data %>% dplyr::select(-cliente_id), type = "prob")[, "Si"]
pred_rf_prob <- predict(rf_final, newdata = test_data %>% dplyr::select(-cliente_id), type = "prob")[, "Si"]
pred_lr_prob <- predict(modelo_lr, newdata = test_lr_final, type = "response")

# Crear objetos ROC
roc_arbol <- pROC::roc(test_data$fugado, pred_arbol_prob, levels = c("No", "Si"))
roc_rf <- pROC::roc(test_data$fugado, pred_rf_prob, levels = c("No", "Si"))
roc_lr <- pROC::roc(test_lr_final$fugado, pred_lr_prob, levels = c("No", "Si"))

# Calcular AUC
auc_arbol <- as.numeric(pROC::auc(roc_arbol))
auc_rf <- as.numeric(pROC::auc(roc_rf))
auc_lr <- as.numeric(pROC::auc(roc_lr))

# Crear gráfico comparativo
plot(roc_arbol, col = "darkgreen", lwd = 2, main = "Comparación de Curvas ROC - Modelos de Churn")
lines(roc_rf, col = "steelblue", lwd = 2)
lines(roc_lr, col = "darkred", lwd = 2)

# Línea de referencia (clasificador aleatorio)
abline(a = 0, b = 1, lty = 2, col = "gray")

# Leyenda
legend("bottomright", 
       legend = c(paste0("Árbol de Decisión (AUC = ", round(auc_arbol, 3), ")"),
                  paste0("Random Forest (AUC = ", round(auc_rf, 3), ")"),
                  paste0("Regresión Logística (AUC = ", round(auc_lr, 3), ")")),
       col = c("darkgreen", "steelblue", "darkred"),
       lwd = 2,
       cex = 0.9)

# Guardar gráfico
ggsave(here("resultados", "curvas_roc_comparativas.png"), 
       width = 10, height = 8, dpi = 300)
cat("✓ Gráfico ROC guardado en: resultados/curvas_roc_comparativas.png\n")

# Mostrar AUCs
cat("\nAUC por modelo:\n")
cat("Árbol de Decisión:", round(auc_arbol, 4), "\n")
cat("Random Forest:", round(auc_rf, 4), "\n")
cat("Regresión Logística:", round(auc_lr, 4), "\n")


# 12. APLICACIÓN DEL MODELO A DATOS ACTUALES REALES -----------------------
# Objetivo: construir el dataset de scoring con datos más recientes, replicando
# los supuestos y preprocesamiento del entrenamiento, y obtener:
# (a) predicción (clase y prob), (b) KPIs de negocio por cliente y
# (c) un resumen ejecutivo de la cohorte actual.

# 12.1 Defino fechas
# Defino un inicio "razonable" para limitar el volumen (abril 2025)
# y la fecha de corte como la máxima observada en ventas (consistente con entrenamiento).

fecha_inicio_actual <- as.Date("2025-04-01")  
fecha_corte_actual <- max(ventas$fecha_venta)  

ventas_actuales <- ventas %>%
  filter(fecha_venta >= fecha_inicio_actual)


# 12.3 Usuarios activos (DEFINICIÓN: últimos 60 días desde la fecha máxima)
# Definición operativa de activo: tuvo al menos 1 venta en los últimos 60 días
# a la fecha de corte. Esta definición ancla el scoring a cuentas “vivas”. 

ventana_churn_actual <- 60
cuentas_activas_actuales <- ventas %>%
  filter(fecha_venta > (fecha_corte_actual - ventana_churn_actual) & 
         fecha_venta <= fecha_corte_actual) %>%
  distinct(cliente_id)

cat("Usuarios activos en últimos 60 días:", nrow(cuentas_activas_actuales), "\n")

# 12.4 Features para todos los datos disponibles
# Replico lógicas de entrenamiento: día de semana, mapeo de canal por producto,
# y flag de aprobación. Mantener mismas categorías asegura compatibilidad del modelo.

ventas_target_actual <- ventas %>%
  filter(fecha_venta >= fecha_inicio_actual) %>%
  mutate(
    dia_semana = wday(fecha_venta, label = FALSE, week_start = 1),
    canal = case_when(
      producto %in% c("mpos", "standalone", "qr", "SDK", "smartpos", "softpos", "caja_pos") ~ "presencial",
      producto %in% c("payment_link", "api_checkout", "plugins", "online_catalog", "api_payments", "PCT", "tienda_geo") ~ "online",
      TRUE ~ "otro"
    ),
    aprobada = ifelse(estado == "APPROVED", 1, 0)
  )

# 12.5 Features históricos por cliente 
# Calculo indicadores de ritmo/variabilidad y mix de canales.
# Nota: para métricas que requieren ≥2 eventos, devolvemos NA y luego se imputa
# igual que en el entrenamiento (consistencia de pipeline).

features_actuales <- ventas_target_actual %>%
  semi_join(cuentas_activas_actuales, by = "cliente_id") %>%
  arrange(cliente_id, fecha_venta) %>%
  group_by(cliente_id) %>%
  summarise(
    dias_entre_ultimas_2_ventas = ifelse(n() >= 2, as.numeric(diff(tail(fecha_venta, 2))), NA),
    variabilidad_entre_ventas = ifelse(n() >= 2, sd(as.numeric(diff(sort(fecha_venta)))), NA),
    frecuencia_promedio_historica = ifelse(n() > 1,
                                         as.numeric((max(fecha_venta) - min(fecha_venta)) / (n() - 1)), NA),
    ticket_promedio = mean(monto_venta, na.rm = TRUE),
    desviacion_ticket = sd(monto_venta, na.rm = TRUE),
    monto_total_historico = sum(monto_venta, na.rm = TRUE),
    mes_ultima_venta = lubridate::month(max(fecha_venta)),
    ventas_en_semana = sum(!dia_semana %in% c(6, 7)),
    ventas_en_fin_de_semana = sum(dia_semana %in% c(6, 7)),
    q_productos = n_distinct(producto),
    ventas_online = any(canal == "online"),
    ventas_presencial = any(canal == "presencial"),
    ventas_mixtas = ventas_online & ventas_presencial,
    total_ventas = n(),
    total_aprobadas = sum(aprobada),
    tasa_aprobacion = ifelse(total_ventas > 0, total_aprobadas / total_ventas, NA),
    usuario_nuevo = min(fecha_venta) >= (fecha_corte_actual - 90),
    .groups = "drop"
  )

# 12.6 Ventanas temporales para datos actuales
# Función auxiliar para generar KPIs por ventana “rolling” y facilitar joins.

ventanas_actuales <- function(d) {
  ventas_target_actual %>%
    filter(fecha_venta > (fecha_corte_actual - d), fecha_venta <= fecha_corte_actual) %>%
    group_by(cliente_id) %>%
    summarise(
      !!paste0("q_ventas_ultimos_", d, "d") := n(),
      !!paste0("monto_total_ultimos_", d, "d") := sum(monto_venta, na.rm = TRUE),
      !!paste0("ticket_prom_ultimos_", d, "d") := mean(monto_venta, na.rm = TRUE),
      .groups = "drop"
    )
}

ventanas_15_actual <- ventanas_actuales(15)
ventanas_30_actual <- ventanas_actuales(30)
ventanas_60_actual <- ventanas_actuales(60)
ventanas_90_actual <- ventanas_actuales(90)

# 12.7 Dataset final para datos actuales
# Ensamblo todas las ventanas.Calculo “tasa_cambio_actividad” y variaciones
# respecto al histórico (señales tempranas de desaceleración/anomalía).

dataset_actual <- features_actuales %>%
  left_join(ventanas_15_actual, by = "cliente_id") %>%
  left_join(ventanas_30_actual, by = "cliente_id") %>%
  left_join(ventanas_60_actual, by = "cliente_id") %>%
  left_join(ventanas_90_actual, by = "cliente_id") %>%
  mutate(
    q_ventas_ultimos_15d = replace_na(q_ventas_ultimos_15d, 0),
    q_ventas_ultimos_30d = replace_na(q_ventas_ultimos_30d, 0),
    q_ventas_ultimos_60d = replace_na(q_ventas_ultimos_60d, 0),
    q_ventas_ultimos_90d = replace_na(q_ventas_ultimos_90d, 0),
    ticket_prom_ultimos_15d = replace_na(ticket_prom_ultimos_15d, 0),
    ticket_prom_ultimos_30d = replace_na(ticket_prom_ultimos_30d, 0),
    ticket_prom_ultimos_60d = replace_na(ticket_prom_ultimos_60d, 0),
    ticket_prom_ultimos_90d = replace_na(ticket_prom_ultimos_90d, 0),
    tasa_cambio_actividad = q_ventas_ultimos_15d / pmax(q_ventas_ultimos_30d / 2, 1),
    variacion_frecuencia = (15 / pmax(q_ventas_ultimos_15d, 1)) - frecuencia_promedio_historica,
    variacion_ticket = (ticket_prom_ultimos_15d - ticket_promedio) / ticket_promedio
  ) %>%
  select(-total_ventas, -total_aprobadas)

# 12.8 Unión con contexto
# Limito a activos y agrego atributos de cuenta. Renombro provincia
# para alinear con el entrenammiento y evitar colisiones de nombres.

dataset_actual <- dataset_actual %>%
  semi_join(cuentas_activas_actuales, by = "cliente_id")

dataset_final_actual <- dataset_actual %>%
  left_join(cuentas, by = "cliente_id") %>%
  rename(provincia = province) %>%
  select(-matches("^province$"))

# 12.9 Imputación de NAs (mismo proceso que para entrenar el modelo)
# Variables de ventanas temporales
vars_ventanas_actual <- c(
  "q_ventas_ultimos_15d","monto_total_ultimos_15d","ticket_prom_ultimos_15d",
  "q_ventas_ultimos_30d","monto_total_ultimos_30d","ticket_prom_ultimos_30d",
  "q_ventas_ultimos_60d","monto_total_ultimos_60d","ticket_prom_ultimos_60d",
  "q_ventas_ultimos_90d","monto_total_ultimos_90d","ticket_prom_ultimos_90d"
)
vars_ventanas_actual <- intersect(vars_ventanas_actual, names(dataset_final_actual))

dataset_final_actual <- dataset_final_actual %>%
  mutate(across(all_of(vars_ventanas_actual), ~ tidyr::replace_na(., 0)))

# Variables que requieren mínimo 2 eventos
vars_dep_2_eventos_actual <- c("variabilidad_entre_ventas",
                              "dias_entre_ultimas_2_ventas",
                              "frecuencia_promedio_historica",
                              "desviacion_ticket",
                              "variacion_frecuencia",
                              "variacion_ticket"
)
vars_dep_2_eventos_actual <- intersect(vars_dep_2_eventos_actual, names(dataset_final_actual))

dataset_final_actual <- dataset_final_actual %>%
  mutate(across(all_of(vars_dep_2_eventos_actual), ~ tidyr::replace_na(., -1)))

# NAs residuales numéricos
dataset_final_actual <- dataset_final_actual %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 12.10 Convierto tipos de variables
dataset_final_actual <- dataset_final_actual %>%
  mutate(across(c(ventas_online, ventas_presencial, ventas_mixtas, usuario_nuevo), as.logical))

dataset_final_actual <- dataset_final_actual %>%
  mutate(
    tipo_cuenta = as.factor(tipo_cuenta),
    provincia = as.factor(provincia)
  )

# 12.11 Elimino duplicados antes de hacer predicciones
dataset_final_actual <- dataset_final_actual %>%
  distinct(cliente_id, .keep_all = TRUE)

# 12.12 Aplico el modelo de árbol de decisión entrenado
# Predicción de clase y probabilidad P(churn="Si"). Mantener el mismo “type”
# y etiqueta de columna que en el ajuste del modelo.


# Preparo datos para predicción (sin cliente_id)
datos_para_prediccion <- dataset_final_actual %>% 
  dplyr::select(-cliente_id)

# Aplico predicciones
predicciones_actual <- predict(arbol_final, newdata = datos_para_prediccion, type = "class")
probabilidades_actual <- predict(arbol_final, newdata = datos_para_prediccion, type = "prob")[, "Si"]

# 12.13 Calculo promedios mensuales históricos
# Calculo promedio de transacciones y volumen mensual (solo APPROVED).

# Verificación del filtro de estado
promedios_mensuales <- ventas %>%
  filter(estado == "APPROVED") %>%
  filter(fecha_venta >= fecha_inicio_actual) %>%
  mutate(mes = floor_date(as.Date(fecha_venta), "month")) %>%
  group_by(cliente_id, mes) %>%
  summarise(
    tx_mes = n(),
    volumen_mes = sum(monto_venta, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(cliente_id) %>%
  summarise(
    meses_activos = n(),
    total_tx = sum(tx_mes),
    total_volumen = sum(volumen_mes),
    promedio_tx_mensual = mean(tx_mes),
    promedio_volumen_mensual = mean(volumen_mes),
    .groups = "drop"
  )

# 12.14 Creao dataframe final 
resultado_final <- dataset_final_actual %>%
  select(cliente_id) %>%
  mutate(
    churn = predicciones_actual,
    probabilidad_churn = probabilidades_actual
  ) %>%
  left_join(promedios_mensuales, by = "cliente_id") %>%
  mutate(
    promedio_tx_mensual = ifelse(is.na(promedio_tx_mensual), 0, promedio_tx_mensual),
    promedio_volumen_mensual = ifelse(is.na(promedio_volumen_mensual), 0, promedio_volumen_mensual)
  ) %>%
  select(
    cliente_id,
    churn,
    probabilidad_churn,
    promedio_tx_mensual,
    promedio_volumen_mensual
  )

# 12.15 Resumen de resultados 
cat("\n=== RESUMEN DE RESULTADOS ===\n")
cat("Total de clientes analizados:", nrow(resultado_final), "\n")
cat("Clientes predichos como churn:", sum(resultado_final$churn == "Si"), "\n")
cat("Porcentaje de churn predicho:", round(mean(resultado_final$churn == "Si") * 100, 2), "%\n")

cat("\nDistribución de probabilidades de churn:\n")
print(summary(resultado_final$probabilidad_churn))

cat("\nPromedio de transacciones mensuales:\n")
print(summary(resultado_final$promedio_tx_mensual))

cat("\nPromedio de volumen mensual (USD):\n")
print(summary(resultado_final$promedio_volumen_mensual))

# 12.15 Guardo resultados df
write.csv(resultado_final,"C:/Users/gimena.velo/Desktop/resultado_final.csv", row.names = FALSE)

# 13. Cálculo del valor actual monto pérdida esperado ---------------------

# 13.1 Parámetros para el cálculo
i <- 0.04  # Tasa de interés anual del 1%
v <- 1 / (1 + i)  # Factor de descuento v = 1/(1+i)
N <- nrow(resultado_final)  # Total de clientes en el dataframe

cat("\n=== CÁLCULO DE VALOR PRESENTE NETO ===\n")
cat("Tasa de interés (i):", i * 100, "%\n")
cat("Factor de descuento (v):", round(v, 6), "\n")
cat("Total de clientes (N):", N, "\n")

# 13.2 Cálculo de la fórmula: ∑(n=1)^N [P_n * M_n * v^(1/2) * (1-v^12)/(1-v)]
# Donde:
# P_n = probabilidad_churn
# M_n = promedio_volumen_mensual
# v = 1/(1+i) = 1/(1+0.04) = 1/1.04

# Calculo v^(1/2) = v^0.5
v_raiz <- v^(1/2)

# Calculo (1-v^12)/(1-v)
factor_anualizacion <- (1 - v^12) / (1 - v)

cat("v^(1/2):", round(v_raiz, 6), "\n")
cat("Factor de anualización (1-v^12)/(1-v):", round(factor_anualizacion, 6), "\n")

# 13.3 Aplicación de la fórmula por cliente
resultado_final <- resultado_final %>%
  mutate(
    # Componente individual: P_n * M_n * v^(1/2) * (1-v^12)/(1-v)
    vpn_individual = probabilidad_churn * promedio_volumen_mensual * v_raiz * factor_anualizacion
  )

# 13.4 Cálculo del VPN total
vpn_total <- sum(resultado_final$vpn_individual, na.rm = TRUE)

cat("\n=== RESULTADOS ===\n")
cat("VPN Total:", round(vpn_total, 2), "USD\n")

