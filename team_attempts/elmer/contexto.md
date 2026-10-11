# BRIEFING PARA AGENTES `eda`, `features` y `modelo`

> **Estado de este documento (03/10, 15:19).** Etapa 1 de 4, el marco de
> metodología. Es el documento que se lee **antes** de empezar, así que su
> encuadre sigue siendo válido y las técnicas que propone se aplicaron todas.
>
> Los Ginis de la §2 son **los que ha medido el equipo en retos anteriores**, no los
> de este reto. El resultado de este reto está en `artifacts/resultado_final.md`:
> **0,23027** en la muestra de reserva, que es el único número comparable con los de
> la §2. La comparación completa, con las condiciones en las que se midieron
> aquellos, está en `artifacts/modelo.md`.

## 1. Encuadre del problema

Según `docs/01_catalogo_problemas_banca.md` (§1 y §5), este reto se encuadra como **propensión de conversión (response/activation)** en banca, con horizonte mensual y un único disparo (sin feedback en tiempo real).

Las siete preguntas clave:

1. **¿Qué queremos predecir?**: Probabilidad de que un cliente alcance el evento `objetivo = 1` en el mes observado (`(id_cliente, mes)`).
2. **¿Unidad de observación?**: **Clave compuesta `(id_cliente, mes)`**. En `train.csv` un cliente aparece en varios meses (panel). En `test.csv` hay **1 fila por cliente** y **todas son `mes = 202612`** (predecir diciembre).
3. **¿Tasa de evento?**: **15,05 %** de positivos (`objetivo`). Tasa estable por mes (14,07%–15,74%).
4. **¿Horizonte y ventana?**: Mensual. Panel de ~4,5 meses por cliente (24.628 clientes únicos en train, 110.100 filas).
5. **¿Leakage/estructura temporal?**: **Crítico**. Según `metaData.csv`: *"un cliente que tiene `objetivo = 1` ya no vuelve a aparecer en meses posteriores"*. Esto implica **dependencia temporal por cliente** y que la aparición futura está truncada por el propio evento.
6. **¿Métrica?**: **Gini = 2 * AUC - 1** (maximizar Gini ≡ maximizar AUC). Implementada en `riesgo_utils.py` (`gini(y_true, y_score)`).
7. **¿Validación válida?**: **Forward-chaining por mes (NO KFold aleatorio)**. Clientes compartidos entre train/val rompen la estructura (y el truncamiento por evento lo agrava). Reserva única: **noviembre (202611)**. Test (202612) **solo lo toca `modelo`**.

**Datos base (confirmados):**
- `train.csv`: 110.100 × 25 | `test.csv`: 9.900 × 24 (sin `objetivo`) | `metaData.csv`: 25 filas | `sample_submission.csv`: 9.900 filas (`id_cliente,prediccion`)
- Sin nulos en los ficheros del reto.
- **Clave no es `id_cliente` solo**: usar `(id_cliente, mes)`. En test sí hay 1 cliente/fila (mes 202612).

## 2. Ginis que ya se han conseguido

La fuente de verdad metodológica son los `docs/` de Templates. Los números de años previos están en sus soluciones. **Si no aparece explícito, se marca "no medido"**.

| Solución / Material | Modelo usado (según fuente) | Nº variables (aprox.) | Gini (reportado) | Referencia |
|---|---|---|---|---|
| **Templates (README)** | Enfoque por embudos + selección iterativa (material de referencia) | 15 (conjunto destacado) | **"no medido" en leaderboard**. Hallazgo clave: conjunto de 15 parecía **peor en validación** y **ese coste desaparecía en la muestra de reserva**. | `contexto/Templates-*/Templates/README.md` |
| **Templates (caso completo)** | Baseline/flujo de referencia (notebooks 00–10) | Variable (flujo) | **no medido** (plantilla orientada a metodología, no a un score fijo) | `contexto/Templates-*/Templates/notebooks/10_caso_completo_kaggle.ipynb` |
| **2024 – 2º puesto** | Solución documentada (2nd_place_solution) | Según README/plancha | **Reportado en material 2024** (ver plancha). Extraer valor exacto si necesario, pero en briefing: **no medido explícito aquí** (leer `2nd_place_solution/README.md` y `Team_Solution/plancha.md`) | `contexto/2024_caso/2nd_place_solution/README.md`, `contexto/2024_caso/Team_Solution/plancha.md` |
| **2024 – Team_Solution** | Preselección + enfoque de equipo | Detallado en `preseleccion.md` | **no medido** (plancha con resultados del equipo) | `contexto/2024_caso/Team_Solution/preseleccion.md`, `plancha.md` |
| **2025 – Team_Solution** | Solución del equipo 2025 | A revisar en carpeta | **no medido** (pendiente de extraer del material) | `contexto/2025_caso/Team_Solution/` |

**Nota metodológica (Templates):** El hallazgo de "validación engañosa vs reserva" es el más relevante para este reto (sin leaderboard). Esto refuerza la necesidad de **media ± desviación** por folds y **reserva única (nov)**.

## 3. Técnicas que funcionaron y que fallaron

Basado en `docs/02_los_dos_embudos.md`, `docs/03_criterios_y_metricas.md`, `README.md` de Templates, CheatSheet y soluciones 2024/2025.

| Técnica | Estado | Qué era | Gini (reportado) | Condiciones / Explicación |
|---|---|---|---|---|
| **Embudo de selección (dos embudos)** | **Funcionó** | Separar: (1) embudo de **disponibilidad/viabilidad** (datos, nulos, estabilidad, cobertura) y (2) embudo de **poder predictivo/valor** (univariante + estabilidad + colinealidad + leakage). | no medido | Filosofía central de Templates. Obliga a justificar cada descarte. Ver `docs/02_los_dos_embudos.md`. |
| **Selección iterativa (hacia adelante / hacia atrás)** con validación por folds temporales | **Funcionó** | Construir conjunto creciente/decreciente midiendo **media ± desv** en folds forward-chaining (no aleatorio). | no medido | Evita sobreajuste a un fold. Templates insiste en **no elegir por un único fold**. |
| **Validación forward-chaining + reserva única (nov)** | **Funcionó (clave)** | folds: `ene-jun->jul`, `ene-jul->ago`, `ene-ago->sep`, `ene-sep->oct`. **Reserva: nov**. Reentrenar `ene-nov` -> predecir dic (test). | no medido | Directamente por truncamiento: cliente con `objetivo=1` no reaparece. Aleatorio **filtra**. Ver AGENTS.md. |
| **Auditoría de fuga (leakage)** | **Funcionó** | Usar `auditar_fuga()` / checklist de `docs/01` §7 (errores que descalifican). | no medido | Especialmente por entidad-cliente y por información futura/derivada post-evento. |
| **Binning/optbinning + monotonicidad / estabilidad (PSI)** | **Funcionó** | Discretización con criterios de negocio/estabilidad. `riesgo_utils.py`: `tabla_eficiencia()`, `psi()`, `validar_calibracion()`. | no medido | Busca robustez temporal (panel). |
| **Importancia + estabilidad entre folds** | **Funcionó** | No quedarse con top puntual: exigir **consistencia** entre folds temporales. | no medido | Reduce riesgo de "suerte" en un fold. |
| **Conjuntos pequeños (15 vars)** vs grandes | **Mixto (hallazgo)** | Templates señala: mejor conjunto de **15 variables** parecía **peor en validación** y **ese coste desaparecía en reserva**. | no medido | **Falló en validación, funcionó/neutral en reserva**. Aprendizaje: no penalizar conjuntos más parsimoniosos si el gap se cierra en reserva temporal. |
| **KFold aleatorio** | **Falló (desaconsejado)** | Dividir filas aleatoriamente ignorando `(id_cliente,mes)` y truncamiento por evento. | — | **Descalifica** para este caso (clientes compartidos train/val). Causa sobreestimación optimista. |
| **Feature engineering sin justificación de negocio** | **Falló** | Crear variables sin anclarse a familias bancarias (RFM, recencia/frecuencia/consistencia monetaria, productos, deuda/ingresos, utilización). | — | Genera ruido y dificulta estabilidad temporal. |
| **Imputación genérica sin cubrir ausencia lógica** | **Falló (cuidado)** | Dataset **sin nulos**. Cualquier imputación debe quedar como **demo/control** (no forzar). | — | Ver notebook `03_preprocesamiento` (tiene restos). |
