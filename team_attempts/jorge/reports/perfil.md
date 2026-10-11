# Perfil preliminar

## Tamaño

| Archivo | Filas | Columnas |
|---|---|---|
| train | 110,100 | 25 |
| test | 9,900 | 24 |

- Llave (id_cliente, mes) duplicada en train: **0**
- Nulos en train: **0**, en test: **0**
- Tasa objetivo global: **0.1505**

## Por mes

| Mes | Filas | Tasa |
|---|---|---|
| 202601 | 10,400 | 0.1462 |
| 202602 | 9,800 | 0.1574 |
| 202603 | 10,300 | 0.1483 |
| 202604 | 9,600 | 0.1534 |
| 202605 | 10,100 | 0.1568 |
| 202606 | 10,350 | 0.1546 |
| 202607 | 9,700 | 0.1454 |
| 202608 | 10,050 | 0.1442 |
| 202609 | 9,900 | 0.1407 |
| 202610 | 10,400 | 0.1565 |
| 202611 | 9,500 | 0.1515 |
| test 202612 | 9,900 | — |

## Estructura de panel

- Clientes en train: **24,628**; meses por cliente: media 4.47, máx 11
- Clientes que convirtieron: **16,567**; filas después de una conversión: **0** (0 = sale del panel al convertir)
- Test que viene del último mes sin convertir: **8,061**; nuevos: **1,839**

## Variables

| Variable | Tipo | Únicos | Gini univariado | % clientes con valor fijo |
|---|---|---|---|---|
| banda_riesgo | object | 3 | +0.1586 | 100% |
| numero_productos | int64 | 5 | +0.1136 | 100% |
| dias_ultima_transaccion | int64 | 364 | -0.0929 | 100% |
| dias_ultima_interaccion | int64 | 364 | -0.0579 | 20% |
| activo_movil | bool | 2 | +0.0396 | 100% |
| tiene_tarjeta_credito | bool | 2 | +0.0224 | 100% |
| canal_adquisicion | object | 5 | +0.0195 | 100% |
| antiguedad_cuenta_meses | int64 | 179 | +0.0175 | 100% |
| ratio_deuda_ingresos | float64 | 24582 | -0.0173 | 100% |
| ocupacion | object | 5 | +0.0135 | 100% |
| saldo_promedio | float64 | 22985 | +0.0131 | 100% |
| edad | int64 | 50 | +0.0112 | 100% |
| region | object | 5 | +0.0100 | 100% |
| ingresos | float64 | 24232 | +0.0091 | 100% |
| tiene_prestamo | bool | 2 | -0.0082 | 100% |
| dispositivo_principal | object | 4 | +0.0043 | 100% |
| visitas_web_ultimos_90_dias | int64 | 23 | -0.0038 | 100% |
| distancia_sucursal_km | float64 | 24588 | +0.0031 | 100% |
| dia_preferido_pago | int64 | 28 | -0.0016 | 100% |
| tiene_seguro | bool | 2 | +0.0015 | 100% |
| es_nuevo_cliente | bool | 2 | -0.0008 | 100% |
| antiguedad_direccion_meses | int64 | 239 | -0.0000 | 100% |

_Gini de categóricas: con la tasa por categoría en train (optimista, solo para ordenar)._

## Categorías nuevas en test

Ninguna
