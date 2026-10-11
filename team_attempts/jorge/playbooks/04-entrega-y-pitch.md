# Playbook · Entrega y pitch

## Cronograma del día del reto
| Momento | Qué pasa | Quién |
|---|---|---|
| H0 – H0:30 | Leer el enunciado, llenar `RETO.md`, ajustar `src/config.py` | Todos |
| H0:30 – H1 | Perfil y **entrega de seguridad** (línea base validada) | DE + ML |
| H1 – H3 | EDA, variables y escalera de modelos en paralelo | DE ↔ ML |
| H2 – H4 | Optimización con las restricciones oficiales o las propuestas | Optimización |
| H3 – H4 | Ensamble, entrega final y congelamiento de cifras | ML |
| H3 – fin | Caso de negocio y presentación (se arma desde la H3, no al final) | Negocio + todos |
| Última hora | Ensayo del pitch con cronómetro, sin tocar código | Todos |

## Lista de chequeo de la entrega
- [ ] `python -m src.submit` termina sin error (filas, orden, columnas, rango, nulos).
- [ ] El nombre y el formato del archivo siguen el enunciado al pie de la letra.
- [ ] Se guardó la copia `outputs/submission_<fecha-hora>.csv`.
- [ ] Las cifras de la presentación coinciden con `reports/` (regla R10).

## Estructura del pitch (5–7 min)
1. **Gancho en soles** (30 s).
2. **3 hallazgos** del dato (1 min 30 s).
3. **Solución**: modelo + optimización, en un diagrama (1 min 30 s).
4. **Valor**: lift, captura y soles contra aleatorio y contra greedy (1 min 30 s).
5. **Producción y riesgos** (1 min).
6. **Cierre**: una frase que el jurado recuerde (15 s).

Preguntas que el jurado suele hacer y que hay que tener respondidas: ¿cómo validaron?, ¿hay
fuga?, ¿qué pasa si cambia el presupuesto?, ¿cómo lo explican a un ejecutivo?, ¿cómo lo
ponen en producción?
