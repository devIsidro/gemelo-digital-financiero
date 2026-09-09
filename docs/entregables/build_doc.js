const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const AZUL = "1F3864";
const AZUL_CLARO = "EAF1FB";
const VERDE = "2E7D32";
const VERDE_CLARO = "E8F5E9";
const GRIS = "444444";

const fases = [
  {
    nombre: "Fase 1 — Kick off y Diseño de Solución",
    semanas: [
      ["Semana 1", "Canvas del proyecto, roles definidos, repositorio Git creado."],
      ["Semana 2", "Catálogo preliminar de KPIs."],
      ["Semana 3", "Diagrama de arquitectura, flujo end-to-end."],
      ["Semana 4", "Docker Compose, repositorio inicial, estructura de carpetas."],
    ],
  },
  {
    nombre: "Fase 2 — Ingesta y Capa Bronze",
    semanas: [
      ["Semana 5", "Data profiling, diccionario de datos."],
      ["Semana 6", "Pipeline de carga inicial."],
      ["Semana 7", "Capa Bronze funcional."],
      ["Semana 8", "DAG de Airflow, logs básicos."],
    ],
  },
  {
    nombre: "Fase 3 — Silver y Calidad de Datos",
    semanas: [
      ["Semana 9", "Jobs de PySpark."],
      ["Semana 10", "Dataset Silver v1."],
      ["Semana 11", "Great Expectations o dbt Tests."],
      ["Semana 12", "Dashboard operativo inicial."],
    ],
  },
  {
    nombre: "Fase 4 — Capa Gold y Analítica",
    semanas: [
      ["Semana 13", "Modelo analítico."],
      ["Semana 14", "Tablas Gold."],
      ["Semana 15", "KPIs calculados (riesgo financiero, exposición, comportamiento transaccional)."],
      ["Semana 16", "Dashboard ejecutivo v1."],
    ],
  },
  {
    nombre: "Fase 5 — Inteligencia Artificial y Simulación",
    semanas: [
      ["Semana 17", "Dataset preparado para modelos."],
      ["Semana 18", "Modelo ML funcional (riesgo crediticio)."],
      ["Semana 19", "Simulador de escenarios (Monte Carlo)."],
      ["Semana 20", "Prototipo de asistente financiero (IA)."],
    ],
  },
  {
    nombre: "Fase 6 — Observabilidad y Hardening",
    semanas: [
      ["Semana 21", "Dashboard de monitoreo."],
      ["Semana 22", "Plataforma estable."],
    ],
  },
  {
    nombre: "Fase 7 — Cierre Ejecutivo",
    semanas: [
      ["Semana 23", "Presentación ejecutiva."],
      ["Semana 24", "Demo integral, documentación final, repositorio completo."],
    ],
  },
];

function faseHeading(nombre) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 120 },
    children: [new TextRun({ text: nombre, bold: true, color: AZUL, size: 26 })],
  });
}

function semanaTable(semanas) {
  return new Table({
    width: { size: 9350, type: WidthType.DXA },
    columnWidths: [1800, 7550],
    rows: semanas.map(([semana, texto]) =>
      new TableRow({
        children: [
          new TableCell({
            width: { size: 1800, type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, fill: AZUL_CLARO },
            margins: { top: 100, bottom: 100, left: 120, right: 120 },
            children: [
              new Paragraph({
                children: [new TextRun({ text: semana, bold: true, color: AZUL, size: 20 })],
              }),
            ],
          }),
          new TableCell({
            width: { size: 7550, type: WidthType.DXA },
            margins: { top: 100, bottom: 100, left: 120, right: 120 },
            children: [
              new Paragraph({
                children: [new TextRun({ text: texto, color: GRIS, size: 20 })],
              }),
            ],
          }),
        ],
      })
    ),
  });
}

function marcadorFinFase2() {
  return new Table({
    width: { size: 9350, type: WidthType.DXA },
    columnWidths: [9350],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: 9350, type: WidthType.DXA },
            shading: { type: ShadingType.CLEAR, fill: VERDE_CLARO },
            margins: { top: 160, bottom: 160, left: 200, right: 200 },
            borders: {
              top: { style: BorderStyle.SINGLE, size: 12, color: VERDE },
              bottom: { style: BorderStyle.SINGLE, size: 12, color: VERDE },
              left: { style: BorderStyle.SINGLE, size: 12, color: VERDE },
              right: { style: BorderStyle.SINGLE, size: 12, color: VERDE },
            },
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  new TextRun({ text: "◀ AVANCE ACTUAL DEL PROYECTO — Fase 2 cerrada, arrancando Fase 3", bold: true, color: VERDE, size: 22 }),
                ],
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 60 },
                children: [
                  new TextRun({
                    text: "Fase 2 validada de punta a punta: arquitectura, Docker Compose, DAG de Airflow corriendo en un entorno real (no solo código), Capa Bronze funcional y diccionario de datos entregado.",
                    color: GRIS,
                    size: 18,
                    italics: true,
                  }),
                ],
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 20 },
                children: [
                  new TextRun({
                    text: "Fase 3 / Semana 9 en progreso: job de PySpark para la Capa Silver ya escrito y probado (deduplicación, validación de montos, normalización), con pruebas automáticas pasando. Sigue: conectarlo a Airflow y correrlo en el entorno Docker.",
                    color: GRIS,
                    size: 18,
                    italics: true,
                  }),
                ],
              }),
            ],
          }),
        ],
      }),
    ],
  });
}

const children = [
  new Paragraph({
    heading: HeadingLevel.TITLE,
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: "Gemelo Digital Financiero", bold: true, color: AZUL, size: 40 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [new TextRun({ text: "Plan de Proyecto — Entregables Semanales", color: GRIS, size: 22 })],
  }),
];

fases.forEach((fase) => {
  children.push(faseHeading(fase.nombre));
  children.push(semanaTable(fase.semanas));
  if (fase.nombre.startsWith("Fase 2")) {
    children.push(new Paragraph({ spacing: { before: 200 }, children: [] }));
    children.push(marcadorFinFase2());
  }
});

const doc = new Document({
  sections: [
    {
      properties: {
        page: { size: { width: 12240, height: 15840 } }, // US Letter
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  require("fs").writeFileSync("plan_proyecto_gemelo_digital_financiero.docx", buffer);
  console.log("done");
});
