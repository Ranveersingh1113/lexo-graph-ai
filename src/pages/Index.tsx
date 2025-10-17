import { Hero } from "@/components/Hero";
import { Features } from "@/components/Features";
import { UploadSection } from "@/components/UploadSection";
import { Architecture } from "@/components/Architecture";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Hero />
      <Features />
      <UploadSection />
      <Architecture />
      
      {/* Footer */}
      <footer className="py-12 px-6 border-t border-border/50 bg-muted/20">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div>
              <h3 className="font-bold text-lg mb-2">
                Intelligent Document Understanding
              </h3>
              <p className="text-sm text-muted-foreground">
                Advanced multilingual document analysis powered by AI
              </p>
            </div>
            <div className="flex gap-8 text-sm text-muted-foreground">
              <a href="#" className="hover:text-primary transition-colors">
                Documentation
              </a>
              <a href="#" className="hover:text-primary transition-colors">
                Research Paper
              </a>
              <a href="#" className="hover:text-primary transition-colors">
                GitHub
              </a>
              <a href="#" className="hover:text-primary transition-colors">
                Contact
              </a>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-border/30 text-center text-sm text-muted-foreground">
            © 2025 Document Understanding System. Built for academic research and production use.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
