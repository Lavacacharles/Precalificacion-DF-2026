# Playbook · Comunicación entre agentes

Los agentes de cada rol no comparten memoria. Todo lo que uno necesita saber del otro está
en `comms/bitacora.jsonl` o en un archivo del repo que la bitácora señala. Si no está
escrito, no pasó.

## Comandos
```bash
python -m src.comms leer --para de                       # lo que me escribieron (incluye "todos")
python -m src.comms enviar --de de --para ml --tipo handoff --ronda 1 \
  --asunto "Familia riesgo_x_vinculacion lista" \
  --msg "Agregué 6 variables en src/features.py. Con lgbm suben +0,004 (3 de 4 folds). Úsalas con --familias ...,riesgo_x_vinc" \
  --artefactos src/features.py,notebooks/03_laboratorio_variables.ipynb --exp lgbm_rxv
python -m src.comms visor                                # comms/visor.html
```

## Tipos de mensaje
| Tipo | Cuándo | Debe incluir |
|---|---|---|
| `handoff` | Entrego algo que otro rol va a usar | Qué, dónde (artefactos), cómo usarlo, evidencia (`--exp`) |
| `pedido` | Necesito algo de otro rol | Qué necesito, para qué y cuándo se considera resuelto |
| `respuesta` | Contesto un pedido | `--responde-a <id>` y el resultado |
| `hallazgo` | Descubrí algo que cambia la estrategia de otro | El dato, el número y la implicancia |
| `bloqueo` | No puedo seguir | Qué intenté, el error y qué necesito para destrabarlo |
| `leccion` | Aprendí algo que vale para el día del reto | La regla propuesta para el AGENTS.md o el playbook |

## Reglas
1. **Mensajes con números.** "Mejoró" no sirve; "+0,004 de Gini medio, 3 de 4 folds, exp `lgbm_rxv`" sí.
2. **Un mensaje, un tema.** El asunto se tiene que entender sin abrir el mensaje.
3. **Siempre `--ronda`.** El coordinador abre y cierra las rondas.
4. **No editar archivos de otro rol.** Se pide por la bitácora. Excepción: agregar una
   familia nueva al final de `src/features.py` (DE) o un modelo nuevo a `src/models.py` (ML),
   siempre que se pruebe que el import funciona antes de avisar.
5. **Al cerrar, siempre `handoff` + al menos una `leccion`.**
