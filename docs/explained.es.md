# El paper, explicado fácil

*Para cualquier persona curiosa, sin matemática ni jerga. Simple, pero sin mentir.*

## Primero: ¿qué es un modelo de lenguaje?

ChatGPT, Gemini y compañía son programas entrenados para una sola cosa: **continuar texto**.
Leen millones de libros y páginas web, y aprenden a predecir qué palabra viene después. De ese
entrenamiento sale algo sorprendente: para continuar bien la frase *"La capital de Italia es…"*,
el programa necesita, de alguna manera, *saber* cuál es la capital de Italia.

Por dentro no hay palabras: hay números. En cada instante, el "estado mental" del modelo es una
lista enorme de números — miles de ellos. Podés imaginarlo como **un punto en un mapa gigante**:
cada pensamiento posible es un lugar distinto del mapa.

## La pregunta del paper

Cuando el modelo procesa *"la capital de Italia"*, ¿cómo guarda ese pensamiento?

- **Opción A:** como un bloque indivisible — "capital-de-Italia" es una sola cosa, distinta e
  independiente de "moneda-de-Italia" o "capital-de-Francia".
- **Opción B:** en **dos piezas separadas** — por un lado *la cosa* (Italia), por otro *la
  pregunta que se le hace* (¿capital de…?).

Parece un detalle técnico, pero es la diferencia entre memorizar cada combinación por separado
y tener un sistema con partes reutilizables — como la diferencia entre memorizar "2+3=5,
2+4=6, 2+5=7…" y saber sumar.

## Lo que encontramos: son dos piezas, y se pueden operar

La respuesta es la **Opción B**, y lo sabemos porque hicimos algo mejor que mirar: **operamos**.

En el mapa de pensamientos del modelo, cada pregunta — *¿capital de…?*, *¿moneda de…?*,
*¿idioma de…?* — resulta ser **una flecha**: una dirección concreta en la que se desplaza el
punto. Y esas flechas se pueden manipular:

- **El experimento estrella:** mientras el modelo piensa *"la moneda de Italia es…"*, le
  restamos la flecha *moneda-de* y le sumamos la flecha *capital-de* — sin tocar la palabra
  "Italia" ni nada más. Su preferencia se invierte por completo: deja de inclinarse por
  *"euro"* y se inclina por *"Roma"*. Funciona con **los 20 pares de preguntas que probamos,
  en 3 modelos distintos** (incluso de empresas diferentes). Y si en lugar de la flecha
  correcta sumamos una flecha aleatoria del mismo tamaño, **no pasa nada** — no es que
  cualquier empujón rompa al modelo.
- **El trasplante:** mejor todavía — tomamos el estado interno completo de un modelo pensando
  *"la capital de Italia"* y lo trasplantamos dentro del mismo modelo mientras piensa *"la
  moneda de Italia"*. El modelo **responde "Roma"**: cambia su respuesta real, no solo su
  preferencia.
- **El ensamblaje (el resultado nuevo):** lo mejor de todo — ni siquiera hace falta copiar ese
  estado. Lo **armamos con tres ingredientes promediados** — una base genérica, una parte
  "Italia" y una parte "capital-de" —, escribimos el pensamiento ensamblado en un solo punto, y
  el modelo responde **"Roma" tan seguido como responde cualquier cosa normalmente** (52%,
  cuando su propia precisión es 53%). Si en cambio ponemos la parte "Francia", responde *París*.
  El pensamiento de verdad está hecho de partes, y las partes alcanzan. (Que antes las flechas
  no lo hicieran *hablar* resultó ser una sobredosis: con un empujón suave — de la fuerza justa —
  el empujón también lo hace hablar.)

## Las flechas son de verdad, no un truco del ejemplo

Tres controles para convencer al escéptico:

1. **Países nunca vistos.** Construimos la flecha *capital-de* usando solo 6 países… y funciona
   perfectamente sobre los otros 6. No memorizó ejemplos: capturó la *operación*.
2. **Otras palabras.** Construimos la flecha con frases tipo *"la moneda de Francia"* y la
   probamos sobre *"el dinero que se usa en Francia"* — otra redacción, mismas flechas, mismo
   resultado. La flecha representa **la pregunta**, no las palabras con que se hace.
3. **Otro modelo.** Repetimos toda la receta en un modelo de otra empresa, con otro
   entrenamiento y otro diccionario interno. Todo se repite.

## El detalle más lindo: la pregunta no es la respuesta

*"¿Qué idioma se habla en Italia?"* y *"¿Cómo se le dice a alguien de Italia?"* tienen la
**misma respuesta**: "italiano". Si el modelo solo guardara respuestas, esos dos pensamientos
serían idénticos por dentro. **No lo son**: son dos flechas distintas. El modelo separa *qué le
están preguntando* de *qué palabra va a decir* — como vos distinguís la pregunta de la
respuesta aunque dos preguntas distintas terminen en la misma palabra.

Los lingüistas tienen un nombre viejo para esta estructura: **declinación**. En latín, *rosa* y
*rosam* son la misma flor cumpliendo roles distintos en la oración — una terminación marca el
rol sin cambiar la cosa. El modelo hace algo parecido: "Italia" es la raíz, y *capital-de* es
la terminación que marca qué rol cumple. Hasta el fenómeno de que dos roles compartan una misma
forma superficial (como acá "italiano") tiene nombre en gramática: *sincretismo*.

## Lo que NO funciona así (y por qué es igual de interesante)

Probamos la misma receta con la aritmética: ¿existe una flecha *sumar*, una flecha
*multiplicar*? **No.** Las "flechas" de las operaciones matemáticas salen enredadas con los
números a los que se aplican y no se pueden trasplantar. Esto encaja con lo que otros
investigadores venían encontrando: estos modelos no hacen matemática con reglas limpias sino
con una **bolsa de trucos memorizados**. Que el método distinga dónde hay estructura limpia y
dónde no, es justamente lo que lo hace creíble.

## ¿Y esto para qué sirve?

Estos sistemas se usan cada vez más y casi nadie sabe **cómo** deciden lo que dicen. Este tipo
de trabajo — se llama *interpretabilidad* — es abrir el capó: entender las piezas internas es
el primer paso para poder auditarlas, corregirlas y confiar (o desconfiar) con fundamento.
Acá mostramos que, al menos para preguntas sobre hechos, hay piezas internas con estructura
clara: una parte para la cosa, una parte para la pregunta, y se pueden separar y trasplantar.

## Lo que este trabajo NO dice

- No dice que el modelo "entienda" ni que "piense" como una persona. Hablamos de geometría:
  puntos y flechas en un espacio de números.
- Probamos 5 tipos de pregunta, en inglés, sobre 12 países, en 3 modelos de tamaño mediano.
  El mundo es más grande que eso, y el paper lo dice.
- Empujar las flechas tiene costo: mientras la operación cambia la pregunta con precisión,
  también perturba otras cosas alrededor. Medimos y reportamos ese costo también.

---

**¿Querés verlo con tus propios ojos?** El [explorador interactivo](explorer.md) te deja mover
las flechas vos misma/o y ver cómo cambia la respuesta. Los detalles completos, con todos los
números y controles, están en el [paper](assets/paper.pdf) y en
[evidencia y controles](robustness.md).
