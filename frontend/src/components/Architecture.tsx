import { Card } from "@/components/ui/card";

const phases = [
  {
    phase: "Phase I",
    title: "Hackathon MVP",
    description: "Rapid baseline implementation using pre-trained models and cloud APIs",
    components: [
      "PP-DocLayout-L for layout detection",
      "Google Cloud Vision API for OCR",
      "LayoutXLM baseline for generative tasks",
      "JSON structured output",
    ],
    color: "bg-blue-500",
  },
  {
    phase: "Phase II",
    title: "Research Application",
    description: "Custom-trained models for state-of-the-art performance",
    components: [
      "Fine-tuned LayoutXLM on augmented data",
      "Custom mixed-script OCR engines",
      "Graph-to-Text innovation for charts",
      "Language-specific optimizations",
    ],
    color: "bg-purple-500",
  },
  {
    phase: "Phase III",
    title: "Production System",
    description: "Scalable web application with enterprise features",
    components: [
      "React/Next.js frontend interface",
      "FastAPI/Flask backend services",
      "Task queue for async processing",
      "Real-time visualization & feedback",
    ],
    color: "bg-green-500",
  },
];

export const Architecture = () => {
  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Three-Phase Architecture
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            A strategic implementation approach from rapid prototyping to
            production-ready research application
          </p>
        </div>

        <div className="space-y-8">
          {phases.map((item, index) => (
            <Card
              key={index}
              className="p-8 hover:shadow-xl transition-all duration-300 border-l-4"
              style={{ borderLeftColor: `hsl(var(--primary))` }}
            >
              <div className="flex flex-col md:flex-row md:items-start gap-6">
                <div className="flex-shrink-0">
                  <div className={`w-16 h-16 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-bold text-lg shadow-lg`}>
                    {index + 1}
                  </div>
                </div>

                <div className="flex-grow">
                  <div className="mb-2">
                    <span className="text-sm font-semibold text-primary uppercase tracking-wider">
                      {item.phase}
                    </span>
                  </div>
                  <h3 className="text-2xl font-bold mb-3">{item.title}</h3>
                  <p className="text-muted-foreground mb-6 leading-relaxed">
                    {item.description}
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {item.components.map((component, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-2 p-3 rounded-lg bg-muted/50 text-sm"
                      >
                        <div className="w-2 h-2 rounded-full bg-primary flex-shrink-0" />
                        <span>{component}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
