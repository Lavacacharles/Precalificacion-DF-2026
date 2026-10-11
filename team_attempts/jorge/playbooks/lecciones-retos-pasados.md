# Lecciones de retos pasados

## DataFest 2025 · BCP "Best Channel to Contact"
Fuente: el reporte técnico y `solution_v3.py` de nuestra solución 2025 (repo aparte).

**El caso:** leads preaprobados de TC, préstamo y crédito vehicular por canal (call center o
agencia). Había que entregar propensión (Gini) y una asignación con 6 restricciones
(consentimiento, montos, clientes "caseros", un lead por cliente-producto, capacidades por
celda).

| Funcionó | Falló o quedó pendiente |
|---|---|
| LightGBM por producto, con dtypes reducidos para 7,8 M filas | **Binarizar** la predicción: el 77,6 % marcado como 1 dejó el Gini en ~0,22 (la v3 lo corrigió) |
| Validación temporal (los últimos 2 meses) | Optimización **greedy** en vez de programación lineal entera |
| Cobertura del 100 % garantizada | Sin calibración de probabilidades para el valor esperado |
| Aplicar las restricciones en orden con un conteo por paso | Tasas del 100 % en préstamo y vehicular (sesgo de selección), detectadas tarde |
| Validaciones automáticas de cada restricción al final | El factor de costo del canal (2,5) quedó fijo, sin sensibilidad |

**Patrón que se repite en los retos del BCP:** propensión + decisión con capacidad
limitada de canales + lenguaje de campañas comerciales. Asumir que vendrá algo parecido.

## Ensayo 2026 · Propensión de conversión (este repo)
- Panel de supervivencia: el cliente sale del panel al convertir.
- Variables fijas por cliente → la historia hay que construirla.
- Señal baja (línea base con Gini ≈ 0,25): cada punto de Gini cuesta, y la estabilidad
  entre folds importa más que un pico.
- Sin restricciones en el enunciado → nosotros proponemos el caso de optimización y eso
  diferencia la entrega.

### Ronda 1 con agentes por rol (03/10/2026)
Caja fuerte = noviembre (t=11), nunca usada para decidir durante la búsqueda.

| Candidato | Gini julio–octubre | Gini caja fuerte |
|---|---|---|
| Referencia `smoke_lgbm` | 0,259 | 0,230 |
| LightGBM afinado con Optuna (`lgbm_opt_s5`) | 0,264 | 0,235 |
| Ensamble logística + LightGBM (`de_ens_logreg_lgbm`) | **0,269** | 0,240 |
| **Logística + reglas de negocio, sin ruido (`de_logreg_reglas_sinruido`)** | 0,268 | **0,247** |

Lecciones adoptadas (con su mensaje de origen en `comms/bitacora.jsonl`):
- **Reglas con umbral antes que tuning** (#13, #19). Cuatro flags (riesgo alto inactivo ≥ 180 días, riesgo bajo con ≥ 3 productos, riesgo bajo digital con tarjeta, RDI ≥ 0,6) valieron +0,027 en la logística; Optuna, +0,004 en LightGBM.
- **El campeón lo decide la caja fuerte** (coordinador). Los folds preferían el ensamble por +0,001; la caja fuerte prefirió la logística por +0,007. Lo complejo se ajusta a los meses de búsqueda.
- **Quitar columnas ruidosas es una familia más** (#14). `dias_ultima_interaccion` es una copia ruidosa de la transacción en el 45 % de las filas.
- **El target encoding no suma con pocas categorías** (#14). Con ≤ 15 celdas, el árbol ya aprende la tasa del segmento.
- **Ensambles con pesos iguales, y solo modelos con correlación < 0,95** (#21). El hazard (correlación 0,976) bajó el ensamble.
- **Optimización: el greedy de referencia debe ser por rentabilidad** (#9). El solver suma 9,6 % sobre él; el resto del valor viene de elegir el canal.
- **Precios sombra con el LP relajado** (#10): 1,4 s contra 60 s del entero, con una brecha < 0,1 %.
- **La incrementalidad es el supuesto que más mueve los soles** (#10): S/ 223 a 555 mil. Hay que presentarla como supuesto y proponer un piloto A/B.
- **Optuna en tandas con sqlite** (#20); ningún comando de más de 8 min sin log.
- **Coordinación entre agentes**: ML cambió `models.py` mientras DE lo usaba y rompió su evaluador. Regla: los cambios de interfaz en `src/` se avisan por la bitácora **antes** de hacerlos.
