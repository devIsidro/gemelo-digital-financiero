const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType,
} = require("docx");
const perfil = require("/tmp/perfil.json");

const AZUL = "1F3864";
const AZUL_CLARO = "EAF1FB";
const NARANJA_CLARO = "FDF3E7";
const NARANJA = "B08900";
const GRIS = "444444";

// Descripciones de negocio por columna (dataset: Credit Card Transactions Fraud Detection, Kaggle)
const descripciones = {
  "Unnamed: 0": "Índice de fila del CSV original (se descarta en la ingesta a Bronze).",
  trans_date_trans_time: "Fecha y hora en que ocurrió la transacción.",
  cc_num: "Número de tarjeta de crédito (identificador del cliente, anonimizado).",
  merchant: "Nombre del comercio donde se realizó la transacción.",
  category: "Categoría de gasto (ej. gas_transport, grocery_pos, misc_pos).",
  amt: "Monto de la transacción en USD.",
  first: "Nombre del titular de la tarjeta.",
  last: "Apellido del titular de la tarjeta.",
  gender: "Género del titular (F/M).",
  street: "Calle del domicilio del titular.",
  city: "Ciudad del domicilio del titular.",
  state: "Estado del domicilio del titular.",
  zip: "Código postal del domicilio del titular.",
  lat: "Latitud del domicilio del titular.",
  long: "Longitud del domicilio del titular.",
  city_pop: "Población de la ciudad del titular.",
  job: "Ocupación declarada del titular.",
  dob: "Fecha de nacimiento del titular.",
  trans_num: "Identificador único de la transacción.",
  unix_time: "Marca de tiempo de la transacción en formato Unix.",
  merch_lat: "Latitud del comercio.",
  merch_long: "Longitud del comercio.",
  is_fraud: "Variable objetivo: 1 si la transacción fue fraudulenta, 0 si fue legítima.",
};

function fmtEjemplos(arr) {
  return arr.join(", ");
}

function fmtRango(col) {
  if (col.min === undefined) return "—";
  return `${col.min} a ${col.max} (media ${col.media})`;
}

const headerRow = new TableRow({
  tableHeader: true,
  children: [
    ["Columna", 1500],
    ["Tipo", 900],
    ["Descripción", 3200],
    ["% nulos", 800],
    ["Cardinalidad", 1000],
    ["Rango / ejemplos", 2450],
  ].map(([texto, ancho]) =>
    new TableCell({
      width: { size: ancho, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: AZUL },
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [
        new Paragraph({
          children: [new TextRun({ text: texto, bold: true, color: "FFFFFF", size: 18 })],
        }),
      ],
    })
  ),
});

const rows = [headerRow].concat(
  perfil.map((col) => {
    const desc = descripciones[col.columna] || "—";
    const rango = col.min !== undefined ? fmtRango(col) : fmtEjemplos(col.ejemplos);
    return new TableRow({
      children: [
        [col.columna, 1500, true],
        [col.dtype, 900, false],
        [desc, 3200, false],
        [`${col.pct_nulos}%`, 800, false],
        [String(col.n_unicos), 1000, false],
        [rango, 2450, false],
      ].map(([texto, ancho, bold]) =>
        new TableCell({
          width: { size: ancho, type: WidthType.DXA },
          margins: { top: 60, bottom: 60, left: 100, right: 100 },
          children: [
            new Paragraph({
              children: [new TextRun({ text: String(texto), bold: !!bold, color: GRIS, size: 16 })],
            }),
          ],
        })
      ),
    });
  })
);

const tabla = new Table({
  width: { size: 9850, type: WidthType.DXA },
  columnWidths: [1500, 900, 3200, 800, 1000, 2450],
  rows,
});

const avisoMuestra = new Table({
  width: { size: 9850, type: WidthType.DXA },
  columnWidths: [9850],
  rows: [
    new TableRow({
      children: [
        new TableCell({
          width: { size: 9850, type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: NARANJA_CLARO },
          margins: { top: 140, bottom: 140, left: 180, right: 180 },
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: "Nota sobre la fuente de datos: ", bold: true, color: NARANJA, size: 19 }),
                new TextRun({
                  text: "este perfil se generó sobre una muestra de 500 filas con el mismo esquema del dataset real de Kaggle (\"Credit Card Transactions Fraud Detection\"), usada para validar el pipeline Bronze de punta a punta. Antes de la demo final hay que sustituirla por el CSV completo descargado de Kaggle y re-correr este mismo profiling.",
                  color: GRIS,
                  size: 19,
                }),
              ],
            }),
          ],
        }),
      ],
    }),
  ],
});

const children = [
  new Paragraph({
    heading: HeadingLevel.TITLE,
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Diccionario de Datos", bold: true, color: AZUL, size: 40 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 40 },
    children: [new TextRun({ text: "Gemelo Digital Financiero — Fase 2, Semana 5", color: GRIS, size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [new TextRun({ text: "Fuente: Credit Card Transactions Fraud Detection (Kaggle)", italics: true, color: GRIS, size: 19 })],
  }),
  avisoMuestra,
  new Paragraph({ spacing: { before: 240 }, children: [] }),
  tabla,
];

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840, orientation: "landscape" },
        },
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  require("fs").writeFileSync("diccionario_datos_transacciones.docx", buffer);
  console.log("done");
});
