# Explorador interactivo

Una vista interactiva, estilo 3Blue1Brown, del resultado operador/operando ("declinación") en
Qwen3-1.7B — leída del modelo real, sin re-ejecutar nada. Tres escenas: la **declinación**
(la nube pasa de organizarse por operando a organizarse por operador a lo largo de la
secuencia), la **inyección** de una dirección de operador (mirá la respuesta flipear), y el
**sincretismo** de `language`/`demonym` con la desinencia pura. Incluye un primer sobre el
concepto de caso gramatical (latín / alemán / japonés) para no-lingüistas — en inglés.

**Cómo manejarlo.** *Escena 1 (Declinación):* arrastrá el slider de "country token" a "query token"
y mirá los mismos 60 puntos reorganizarse — mezclados por país a la izquierda, cinco grupos limpios de
preguntas a la derecha; ▶ lo anima. *Escena 2 (Inyección):* elegí un par from→to, tocá **inject**, y
mirá el estado deslizarse al grupo objetivo mientras el margen de respuesta se invierte; después activá
**random control** y mirá cómo una flecha aleatoria igual de grande no hace nada. *Escena 3
(Sincretismo):* dos preguntas que comparten palabra de respuesta viven separadas como direcciones;
disparó la **desinencia pura** y un marcador con la palabra cancelada instala la relación igual.

<iframe src="../../interactive/declension.html" title="Explorador de declinación"
        style="width:100%;height:1180px;border:1px solid #ddd;border-radius:12px"
        loading="lazy"></iframe>

> Página completa: [abrir el explorador directo](../../interactive/declension.html). Enlaces
> directos: [declinación](../../interactive/declension.html#0) ·
> [inyección](../../interactive/declension.html#1) ·
> [sincretismo](../../interactive/declension.html#2).
