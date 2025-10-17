import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, FileText, Image, File, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export const UploadSection = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const { toast } = useToast();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  const handleFiles = (newFiles: File[]) => {
    const validTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    const validFiles = newFiles.filter((file) => {
      if (!validTypes.includes(file.type)) {
        toast({
          title: "Invalid file type",
          description: `${file.name} is not a supported format`,
          variant: "destructive",
        });
        return false;
      }
      return true;
    });

    setFiles((prev) => [...prev, ...validFiles]);
    
    if (validFiles.length > 0) {
      toast({
        title: "Files added",
        description: `${validFiles.length} file(s) ready for processing`,
      });
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (type: string) => {
    if (type.includes("pdf")) return <FileText className="w-5 h-5 text-destructive" />;
    if (type.includes("image")) return <Image className="w-5 h-5 text-primary" />;
    return <File className="w-5 h-5 text-muted-foreground" />;
  };

  const processFiles = () => {
    toast({
      title: "Processing started",
      description: "Your documents are being analyzed...",
    });
    // Processing logic will be implemented in later phases
  };

  return (
    <section id="upload-section" className="py-24 px-6">
      <div className="container mx-auto max-w-5xl">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4">Upload Your Documents</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Support for PDF, DOCX, JPEG, PNG, and WebP formats. Drag and drop or click to browse.
          </p>
        </div>

        <Card
          className={`p-12 border-2 border-dashed transition-all duration-300 ${
            isDragging
              ? "border-primary bg-primary/5 shadow-lg"
              : "border-border hover:border-primary/50"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center justify-center text-center">
            <div className={`w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-6 transition-transform duration-300 ${isDragging ? "scale-110" : ""}`}>
              <Upload className="w-10 h-10 text-primary" />
            </div>

            <h3 className="text-2xl font-semibold mb-2">Drop your files here</h3>
            <p className="text-muted-foreground mb-6">or click to browse from your device</p>

            <input
              type="file"
              multiple
              accept=".pdf,.docx,.jpg,.jpeg,.png,.webp"
              onChange={handleFileInput}
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input">
              <Button variant="default" size="lg" asChild>
                <span className="cursor-pointer">
                  Browse Files
                </span>
              </Button>
            </label>
          </div>
        </Card>

        {/* File list */}
        {files.length > 0 && (
          <div className="mt-8 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">Selected Files ({files.length})</h3>
              <Button onClick={processFiles} variant="hero" size="lg">
                Process Documents
              </Button>
            </div>

            <div className="grid gap-3">
              {files.map((file, index) => (
                <Card key={index} className="p-4 flex items-center justify-between hover:shadow-md transition-shadow">
                  <div className="flex items-center gap-3">
                    {getFileIcon(file.type)}
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeFile(index)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
