# Conocimiento de banca para el reto

## Glosario
| Término | Qué significa | Cómo aparece en los datos |
|---|---|---|
| Propensión | Probabilidad de que el cliente acepte una oferta | La predicción |
| Lead | Cliente + oferta (+ canal) listo para contactar | Una fila del test |
| Campaña | Contacto masivo mensual con ofertas | El mes del test |
| Venta cruzada | Vender un producto nuevo a un cliente actual | `numero_productos`, flags `tiene_*` |
| Vinculación / share of wallet | Cuántos productos tiene con el banco | Suma de flags y productos |
| Banda de riesgo | Probabilidad de impago según el score de riesgo | `banda_riesgo` |
| Apetito de riesgo | Cuánto riesgo acepta el banco en una campaña | Restricción C5 |
| RDI (ratio deuda-ingreso) | Deuda / ingreso; con valores altos la capacidad de pago cae | `ratio_deuda_ingresos` |
| Recencia | Días desde la última transacción o interacción | `dias_ultima_*` |
| Costo por venta | Costo de la campaña / conversiones | Caso de negocio |
| Lift | Tasa del segmento / tasa base | Por decil |
| SBS | Superintendencia de Banca, Seguros y AFP: regula los modelos y la conducta de mercado | Explicabilidad y equidad |

## Intuiciones que conviene verificar en el dato
- Más vinculación → más propensión a comprar otro producto (venta cruzada).
- Riesgo alto → menos oferta aprobada y menos conversión.
- Cliente digital activo → convierte por canales baratos (app).
- Recencia baja (transacción reciente) → cliente activo → más propensión.
- Diciembre: gratificación, consumo de fin de año y más demanda de crédito y tarjeta.
- Clientes nuevos: la primera oferta en los primeros meses suele convertir más.

## Canales típicos y su lógica (supuestos para la optimización)
| Canal | Costo relativo | Efectividad | Cuándo usarlo |
|---|---|---|---|
| App / push digital | Muy bajo | Baja | Cliente con `activo_movil`, cualquier propensión aceptable |
| Call center | Medio | Media | Propensión media-alta, sin exigencia de canal |
| Asesor en agencia | Alto | Alta | Propensión alta y valor alto, sucursal cercana |
