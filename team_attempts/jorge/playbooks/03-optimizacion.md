# Playbook · Optimización

**Misión:** convertir probabilidades en **decisiones**: a quién contactar, por qué canal y
con qué presupuesto, maximizando el valor esperado para el banco.

> En el ensayo 2026 el enunciado no trae restricciones. Las proponemos nosotros como
> **supuestos explícitos y ajustables** en `src/config.py::OPT`. El día del reto se
> reemplazan por las oficiales, si las hay, y se conserva el mismo motor.

## 1. Formulación
Variables: `x[i,c] ∈ {0,1}`, contactar al cliente `i` por el canal `c`.

**Maximizar** el valor esperado neto:

```
Σ x[i,c] · ( p_i · efectividad_c · valor_i  −  costo_c )
```

- `p_i`: probabilidad **calibrada** del modelo.
- `efectividad_c`: cuánto multiplica el canal la conversión (supuesto).
- `valor_i`: margen esperado si convierte = f(ingresos, banda de riesgo).
- `costo_c`: costo por contacto del canal.

## 2. Restricciones propuestas (con su porqué de negocio)
| # | Restricción | Porqué |
|---|---|---|
| C1 | Un cliente recibe como máximo **un** contacto en el mes | Fatiga y experiencia del cliente. |
| C2 | **Capacidad por canal**: call center N₁, asesor en agencia N₂; la app no tiene tope | Los equipos comerciales tienen cupos reales. |
| C3 | **Presupuesto** total de la campaña | Así se decide en la vida real. |
| C4 | **Elegibilidad por canal**: app solo si `activo_movil`; asesor solo si la sucursal está a ≤ X km | No se contacta por un canal que el cliente no usa. |
| C5 | **Apetito de riesgo**: como máximo el Y % de los contactados en banda `high` | Política de crédito responsable. |
| C6 | **Política de contacto**: no contactar a quien tuvo una interacción hace < Z días | Respeto al cliente y a la regulación de protección al consumidor. |
| C7 | Contactar solo si el valor esperado neto es > 0 | No gastar en contactos que destruyen valor. |

## 3. Motor
- **OR-Tools** (solver lineal entero SCIP, o CP-SAT). Con ~10.000 clientes × 3 canales se
  resuelve en segundos.
- Siempre se compara contra:
  1. **Aleatorio** con el mismo presupuesto (lo que pasa sin modelo).
  2. **Greedy**: los top-N por probabilidad, por el canal más barato (lo que hace un
     analista sin optimizar).
  3. **Óptimo** (OR-Tools).
- La diferencia óptimo − greedy es el **valor de la optimización**, y la diferencia
  greedy − aleatorio es el **valor del modelo**. Las dos van al pitch.

## 4. Análisis de sensibilidad (lo que convence al jurado)
- Curva **presupuesto → valor**: dónde está el rendimiento decreciente.
- ¿Qué pasa si el call center tiene 50 % menos cupos?
- ¿Qué pasa si el apetito de riesgo se endurece?
- **Precio sombra** de cada restricción: cuánto valor da un cupo más de call center.

## 5. Entregables
- `outputs/asignacion.csv`: `id_cliente, canal, p, valor_esperado`.
- `reports/optimizacion.md`: supuestos, comparación de las 3 estrategias, sensibilidad y
  la sección "¿Qué significa para el banco?".
