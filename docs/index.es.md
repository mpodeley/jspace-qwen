# Las operaciones relacionales se factorizan de sus operandos en LLMs

<p class="hero-sub">Intervenciones causales revelan <b>direcciones de operador transferibles</b>
para relaciones factuales — replicado en <b>Qwen3-1.7B/8B y Gemma-2-9B</b> — mientras que la
aritmética y la lógica permanecen entrelazadas con sus operandos.</p>

<div class="stats" markdown>
<div class="stat"><span class="n">20/20 × 3</span><span class="d">swaps de operador cambian la
respuesta, en los tres modelos — controles aleatorios de norma igualada ≈ 0</span></div>
<div class="stat"><span class="n">82–86%</span><span class="d">de la varianza del workspace en el
token de consulta es el operador; ~7–13% interacción ("fusión")</span></div>
<div class="stat"><span class="n">180/180</span><span class="d">swaps flipean con operandos
held-out y paráfrasis nunca vistas — un operador real, no interpolación</span></div>
</div>

<div class="hero-buttons" markdown>
[:material-play-circle: Resultado interactivo](explorer.md){ .md-button .md-button--primary }
[:material-file-document: Paper (PDF)](assets/paper.pdf){ .md-button }
[:material-flask: Evidencia y controles](robustness.md){ .md-button }
[:material-github: Código](https://github.com/mpodeley/jspace-qwen){ .md-button }
</div>

<div class="hero-gif" markdown>
[![La declinación: los mismos 60 estados del workspace, organizados por operando en el token del país, se reorganizan por operador en el token de consulta](figs/declension.gif)](explorer.md)
</div>

## La idea en 60 segundos

El latín marca el *rol* de un sustantivo con una desinencia de caso: `ros-a / ros-am` — misma
raíz, distinta función. Encontramos que los LLMs hacen algo estructuralmente similar con las
relaciones factuales. Ante *"The currency of Italy is"*, a mitad de la red el estado de la
oración se describe bien como

```
estado ≈ μ + operando(Italia) + operador(moneda-de) + interacción pequeña
```

La **parte del operador es un objeto causal real**: sumá `v(capital) − v(moneda)` al residual
stream y el modelo responde *Roma* en vez de *euro* — para **todos** los pares ordenados de las
cinco relaciones, en **los tres modelos**. La dirección **transfiere**: construida con la mitad
de los países, flipea la otra mitad; construida con una redacción, flipea paráfrasis nuevas sin
cambios. Y la operación **no es su palabra de salida**: *idioma-de* y *gentilicio-de* producen
ambas "Italian", pero son direcciones distintas — el modelo separa el rol gramatical de la
palabra que lo realiza, como la declinación separa el caso de la forma superficial
(*sincretismo*).

La factorización es **específica de la recuperación relacional**: el mismo pipeline sobre
aritmética (+, ×, −) o lógica de comparación da un operador entrelazado con sus operandos
(2–4× la interacción) que no generaliza — consistente con "la aritmética como bolsa de
heurísticas". [Exploralo interactivo](explorer.md), [leé el paper](assets/paper.pdf), o
[verificá cada claim contra su control](robustness.md).

!!! note "Nota metodológica — el null honesto que originó esto"
    Este proyecto empezó como una réplica del claim de legibilidad del J-space/global
    workspace. Bajo controles apareados (incluida una proyección aleatoria de espectro
    igualado), el **readout del J-lens no superó al logit lens** en ninguna de nuestras
    métricas. La estructura operador/operando de arriba es organización *causal*, no un
    subespacio privilegiadamente legible — y la mitad negativa está documentada con el mismo
    rigor en el [archivo](method.md) y la [bitácora](findings.md).

---

**Reproducibilidad.** Todo corre en una sola APU AMD Strix Halo (sin CUDA). Semillas,
checkpoints exactos (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `google/gemma-2-9b`), artefactos
long-form por operando y el generador de cada figura están en el repo — ver
[Cómo reproducir](reproduce.md). Licencia MIT · Matias Podeley · <mpodeley@gmail.com> ·
[Cómo citar](https://github.com/mpodeley/jspace-qwen#how-to-cite)
