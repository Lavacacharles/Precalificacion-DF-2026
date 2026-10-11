"""Contrasta hipótesis del origen del panel sin modificar datos ni entrenar modelos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp, chi2_contingency

OUT = Path('auditoria_origen_panel')
OUT.mkdir(exist_ok=True)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
d = pd.concat([train, test.assign(objetivo=np.nan)], ignore_index=True)
d['ord'] = d.mes // 100 * 12 + d.mes % 100
d = d.sort_values(['id_cliente', 'ord']).reset_index(drop=True)
features = [c for c in test if c not in ['id_cliente', 'mes']]
static = [c for c in features if c != 'dias_ultima_interaccion']
g = d.groupby('id_cliente', sort=False)
life = g.agg(inicio=('ord', 'min'), fin=('ord', 'max'), n=('mes', 'size'),
             evento=('objetivo', 'sum'), cuenta=('antiguedad_cuenta_meses', 'first'),
             direccion=('antiguedad_direccion_meses', 'first'), edad=('edad', 'first'),
             transaccion=('dias_ultima_transaccion', 'first'), nuevo=('es_nuevo_cliente', 'first'))
life['span'] = life.fin-life.inicio
life['cuenta_retro_inicial'] = life.cuenta-life.span
life['direccion_retro_inicial'] = life.direccion-life.span
multi = life.n.gt(1)
checks = {}
checks['panel'] = dict(filas=len(d), clientes=len(life), clientes_repetidos=int(multi.sum()),
                      pares=int(len(d)-len(life)), huecos=int((life.span+1-life.n).sum()),
                      congeladas=[c for c in features if g[c].nunique().max()==1])

# La hipótesis de snapshot final tiene una implicación verificable para la MISMA cuenta:
# su edad al principio no puede ser negativa (meses enteros, cero permitido).
rows = []
for label, mask in [('todos_repetidos', multi), ('evento_conocido_repetidos', multi & life.evento.eq(1)),
                    ('sin_evento_repetidos', multi & life.evento.eq(0))]:
    z = life.loc[mask]
    rows.append(dict(grupo=label, clientes=len(z),
                     cuenta_inicial_negativa=int(z.cuenta_retro_inicial.lt(0).sum()),
                     cuenta_incompatible_incluso_margen_un_mes=int(z.cuenta_retro_inicial.lt(-1).sum()),
                     cuenta_inicial_cero=int(z.cuenta_retro_inicial.eq(0).sum()),
                     direccion_inicial_negativa=int(z.direccion_retro_inicial.lt(0).sum())))
snapshot = pd.DataFrame(rows)
snapshot.to_csv(OUT/'snapshot_final.csv', index=False)
examples = life.loc[multi & life.cuenta_retro_inicial.lt(0)].head(8)
examples.to_csv(OUT/'contraejemplos_snapshot_final.csv')
checks['snapshot_final'] = rows
checks['ejemplos'] = examples.reset_index().to_dict('records')

# Una recencia congelada implica una fecha de transacción móvil: indicar coincidencia exacta
# de perfiles decimales distingue congelación de simples redondeos en variables enteras.
first = d.drop_duplicates('id_cliente').copy()
last = d.drop_duplicates('id_cliente', keep='last').copy()
checks['perfiles'] = dict(
    perfiles_estaticos_unicos=len(first.drop_duplicates(static)),
    clientes=len(first),
    fechas_implicitas_distintas_para_transaccion=int((life.n>1).sum()),
    clientes_sin_nuevos_cambios_en_todo_el_ano=int(life.n.eq(12).sum()),
    nuevos_con_cuenta_mas_12=int((first.es_nuevo_cliente & first.antiguedad_cuenta_meses.gt(12)).sum()),
    nuevos_total=int(first.es_nuevo_cliente.sum()))

# Uniformidad descriptiva a nivel CLIENTE, no 120000 filas dependientes.
bound_rows = []
for c in ['edad', 'ingresos', 'ratio_deuda_ingresos', 'saldo_promedio', 'distancia_sucursal_km']:
    z=first[c]
    bound_rows.append(dict(variable=c, minimo=float(z.min()), maximo=float(z.max()),
                           clientes_min=int(z.eq(z.min()).sum()), clientes_max=int(z.eq(z.max()).sum())))
checks['limites_por_cliente']=bound_rows

# IDs y entradas: comprobar si el panel se obtiene eliminando eventos y anexando IDs nuevos.
flow=[]
previous=set(); previous_events=set(); running_max=0
for month, z in d.groupby('mes', sort=True):
    now=set(z.id_cliente)
    newcomers=now-previous
    expected=set(range(running_max+1, int(z.id_cliente.max())+1))
    flow.append(dict(mes=int(month), n=len(z), entrantes=len(newcomers),
                     entradas_ids_contiguos=bool(newcomers==expected),
                     continuadores_exactos=bool((now-newcomers)==(previous-previous_events))))
    previous=now
    previous_events=set(z.loc[z.objetivo.eq(1), 'id_cliente'])
    running_max=max(running_max,int(z.id_cliente.max()))
checks['flujo']=flow

# Ruido progresivo: mecanismo candidato conserva original con prob 1-p y reemplaza con p.
# Para un reemplazo independiente de la misma distribución, P(igual)=1-p+p*sum(f_j^2).
rec=[]
for month,z in d.groupby('mes',sort=True):
    p=(int(month)%100-1)/11
    freq=z.dias_ultima_transaccion.value_counts(normalize=True)
    chance=float((freq**2).sum())
    eq=z.dias_ultima_interaccion.eq(z.dias_ultima_transaccion)
    mismatch=(~eq)
    n_noise=round(len(z)*p)
    # La selección de filas ruidosas puede ser aleatoria o concentrarse por orden del archivo.
    sorted_z=z.sort_values('id_cliente')
    sorted_mismatch=sorted_z.dias_ultima_interaccion.ne(sorted_z.dias_ultima_transaccion).to_numpy()
    deciles=pd.qcut(np.arange(len(z)),10,labels=False)
    rates=pd.Series(sorted_mismatch).groupby(deciles).mean()
    rec.append(dict(mes=int(month), n=len(z), p_ruido_candidato=p,
                    igualdad_observada=float(eq.mean()), igualdad_esperada_mezcla=1-p+p*chance,
                    diferencia_igualdad=float(eq.mean()-(1-p+p*chance)),
                    correlacion=float(z.dias_ultima_interaccion.corr(z.dias_ultima_transaccion)),
                    desacuerdos=int(mismatch.sum()), reemplazos_candidatos=n_noise,
                    igualdad_residual_en_reemplazos=n_noise-int(mismatch.sum()),
                    mismos_valores_ordenados=bool(np.array_equal(np.sort(z.dias_ultima_interaccion),np.sort(z.dias_ultima_transaccion))),
                    media_tx=float(z.dias_ultima_transaccion.mean()), media_inter=float(z.dias_ultima_interaccion.mean()),
                    KS_recencias=float(ks_2samp(z.dias_ultima_transaccion,z.dias_ultima_interaccion).statistic),
                    minimo_desacuerdo_decil_id=float(rates.min()),maximo_desacuerdo_decil_id=float(rates.max())))
rec=pd.DataFrame(rec)
rec.to_csv(OUT/'ruido_recencias_por_mes.csv', index=False)
checks['ruido_recencias']=rec.to_dict('records')

# Ajustar mezcla es diagnóstico descriptivo post-hoc, no una prueba de la causa del generador.
x=rec.p_ruido_candidato.to_numpy()
y=rec.igualdad_observada.to_numpy()
beta=np.polyfit(x,y,1)
pred=np.polyval(beta,x)
checks['ajuste_lineal_igualdad']=dict(pendiente=float(beta[0]),intercepto=float(beta[1]),
    R2=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()),
    error_max_mezcla_puntos_porcentuales=float((rec.igualdad_observada-rec.igualdad_esperada_mezcla).abs().max()*100))

# A igual mes, separar primera observación de recurrentes: ¿reloj de calendario o de usuario?
d['primera_obs']=g.cumcount().eq(0)
d['igual_recencia']=d.dias_ultima_interaccion.eq(d.dias_ultima_transaccion)
coh=d.groupby(['mes','primera_obs']).igual_recencia.agg(['mean','size']).reset_index()
coh.to_csv(OUT/'igualdad_por_mes_y_entrada.csv',index=False)
checks['igualdad_por_entrada']=coh.to_dict('records')

# Restricciones de un contador de recencia: no puede subir más que días de calendario sin revisión.
date=pd.to_datetime(d.mes.astype(str),format='%Y%m')
days=date.groupby(d.id_cliente).diff().dt.days
delta=g.dias_ultima_interaccion.diff()
checks['recencia_fisica']=dict(pares=int(days.notna().sum()),
    incrementos_mayores_que_dias_transcurridos=int((delta>days).sum()),
    incrementos_mayores_62=int(delta.gt(62).sum()),
    max_incremento=float(delta.max()), max_descenso=float(delta.min()))

# Correlación edad de cuenta / seguimiento: evidencia débil, confundida por riesgo y selección.
checks['correlaciones_clientes']={c:float(spearmanr(life[c],life.span).statistic)
                                for c in ['cuenta','direccion','edad']}
(OUT/'evidencia.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if k not in ['ruido_recencias','flujo','igualdad_por_entrada']},indent=2,ensure_ascii=False))
