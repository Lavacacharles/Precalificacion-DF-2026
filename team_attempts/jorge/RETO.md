# Ficha del reto — Ensayo DataFest 2026 (propensión de conversión)

> Esta ficha se llena en la hora cero de cada reto. El día final se reemplaza su contenido;
> la estructura se mantiene.

## Enunciado en una línea
Estimar la probabilidad de que cada cliente **convierta por primera vez** en diciembre de 2026.

## Datos
| Archivo | Filas | Columnas | Notas |
|---|---|---|---|
| `train.csv` | 110.100 | 25 | Enero a noviembre de 2026, una fila por cliente-mes |
| `test.csv` | 9.900 | 24 | Diciembre de 2026, sin `objetivo` |
| `sample_submission.csv` | 9.900 | 2 | `id_cliente,prediccion` |
| `metaData.csv` | — | — | Diccionario de columnas |

- **Unidad de análisis:** cliente-mes (`id_cliente`, `mes` en formato AAAAMM).
- **Objetivo:** `objetivo` = 1 si es la **primera** conversión del cliente en ese mes.
- **Métrica oficial:** Gini = 2·AUC − 1. Solo importa el orden → se entrega probabilidad (R1).
- **Entrega:** `id_cliente,prediccion`, mismo orden que `test.csv`, sin la columna `mes`.

## Estructura descubierta (perfil inicial, 03/10/2026)
- **Es un panel de supervivencia:** cuando un cliente convierte, sale del panel. Ningún
  cliente sale sin convertir antes de noviembre. Los meses de cada cliente son consecutivos.
- **El test = 8.061 clientes de noviembre que no convirtieron + 1.839 clientes nuevos.**
- Tasa de conversión ≈ 15 % por mes, estable de enero a noviembre.
- **Las variables son fijas por cliente**, salvo `dias_ultima_interaccion`. La señal que
  cambia en el tiempo tiene que construirse: meses en el panel, cohorte de entrada, cambios
  en la interacción.
- Clientes nuevos por mes: 10.400 en enero y luego entre 700 y 2.000.
- Las variables más fuertes por sí solas: `numero_productos`, `banda_riesgo`,
  `dias_ultima_transaccion`, `activo_movil`, `tiene_tarjeta_credito`.
- `dias_ultima_transaccion` = `dias_ultima_interaccion` en el 55 % de las filas (a investigar).
- Sin nulos (confirmado por el enunciado).

## Línea base
- LightGBM con las variables tal cual, entrenando con enero a septiembre y validando con
  octubre y noviembre: **Gini ≈ 0,25**.

## Restricciones de negocio
El enunciado **no trae restricciones**. Las proponemos nosotros (ver
[`playbooks/03-optimizacion.md`](playbooks/03-optimizacion.md)) y se presentan como
**supuestos explícitos y ajustables**.

## Supuestos abiertos
- S1: Una "conversión" es la contratación de un producto nuevo ofrecido en campaña.
- S2: El banco contacta una vez al mes por cliente y por un solo canal.
- S3: Los costos y márgenes por canal de `config.OPT` son referenciales.
