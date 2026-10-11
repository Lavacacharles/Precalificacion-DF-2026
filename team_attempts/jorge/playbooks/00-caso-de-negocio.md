# Playbook · Caso de negocio

> El modelo no gana el reto. Lo gana **la decisión mejor defendida**. Este playbook lo usan
> todos los roles y tiene un dueño que consolida `reports/negocio.md`.

## 1. La pregunta del banco
"Con un presupuesto limitado de contactos, ¿a quién llamo este mes, por qué canal y cuánto
gano frente a como lo hago hoy?"

## 2. Traducir la métrica a negocio
| Métrica técnica | Cómo se dice en el pitch |
|---|---|
| Gini 0,30 | "Ordenamos a los clientes mucho mejor que el azar: el 20 % de arriba concentra el X % de las conversiones." |
| Lift en el decil 1 | "Contactar al primer decil convierte X veces más que contactar al azar." |
| Curva de captura | "Con el 30 % de los contactos capturamos el Y % de las ventas." |
| Valor óptimo − greedy | "La optimización agrega S/ Z por campaña sin aumentar el presupuesto." |
| Costo por conversión | "Cada venta cuesta S/ A con el modelo, contra S/ B hoy." |

## 3. Cifras obligatorias (salen de `src/business.py`)
- Tasa base de conversión y conversiones esperadas con contacto aleatorio.
- Lift y captura acumulada por decil (validación temporal, no test).
- Valor esperado de las 3 estrategias (aleatorio, greedy y óptimo) con el mismo presupuesto.
- Proyección anual: valor por campaña × 12, **etiquetada como estimación**.

## 4. Hilo narrativo con guiños al BCP y a ESAN
- **BCP**: el banco más grande del país, con una base masiva y digital (Yape, banca móvil).
  El reto es priorizar en esa escala, no convencer a pocos.
- **ESAN**: rigor de gestión. Cada recomendación con supuesto, costo, beneficio y riesgo.
- Lenguaje del banco: *propensión*, *campaña*, *canal*, *apetito de riesgo*,
  *venta cruzada*, *vinculación*, *costo por venta*.

## 5. Riesgos y producción (siempre se presentan)
- Deriva del modelo → monitoreo mensual del PSI y del Gini, y reentrenamiento mensual.
- Sesgos → revisar el desempeño por región, edad y ocupación (equidad).
- Explicabilidad → SHAP por cliente para el ejecutivo comercial.
- Lo que no sabemos → efecto causal del contacto (no hay grupo de control); se propone un
  piloto A/B para medir el *uplift* real.
