# PE extraction code Written by: Ajit kumar, urwithajit9@gmail.com ,25Feb2015

import os
import pefile
import csv

rootdir = "/home/ubuntu/restless/restless/data"

# Change this based on the PE file we're analyzing -
# [0] for benign and [1] for malicious, or leave empty if we do$
class_label = []

IMAGE_DOS_HEADER = [
    "e_magic",
    "e_cblp",
    "e_cp",
    "e_crlc",
    "e_cparhdr",
    "e_minalloc",
    "e_maxalloc",
    "e_ss",
    "e_sp",
    "e_csum",
    "e_ip",
    "e_cs",
    "e_lfarlc",
    "e_ovno",
    "e_res",
    "e_oemid",
    "e_oeminfo",
    "e_res2",
    "e_lfanew",
]

FILE_HEADER = [
    "Machine",
    "NumberOfSections",
    "CreationYear",
    "PointerToSymbolTable",
    "NumberOfSymbols",
    "SizeOfOptionalHeader",
    "Characteristics",
]

OPTIONAL_HEADER = [
    "Magic",
    "MajorLinkerVersion",
    "MinorLinkerVersion",
    "SizeOfCode",
    "SizeOfInitializedData",
    "SizeOfUninitializedData",
    "AddressOfEntryPoint",
    "BaseOfCode",
    "BaseOfData",
    "ImageBase",
    "SectionAlignment",
    "FileAlignment",
    "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion",
    "MajorImageVersion",
    "MinorImageVersion",
    "MajorSubsystemVersion",
    "MinorSubsystemVersion",
    "SizeOfImage",
    "SizeOfHeaders",
    "CheckSum",
    "Subsystem",
    "DllCharacteristics",
    "SizeOfStackReserve",
    "SizeOfStackCommit",
    "SizeOfHeapReserve",
    "SizeOfHeapCommit",
    "LoaderFlags",
    "NumberOfRvaAndSizes",
]


class PE_Analyzer:
    """
    Contains tools for analyzing and extracting PE file headers and other metadata.
    """

    def __init__(self):
        self.class_label = class_label
        self.rootdir = rootdir

    def file_creation_year(self, seconds):
        return 1970 + ((int(seconds) / 86400) / 365)

    def extract_dos_header(self, pe):
        IMAGE_DOS_HEADER_data = [0 for i in range(19)]
        IMAGE_DOS_HEADER_data = [
            pe.DOS_HEADER.e_magic,
            pe.DOS_HEADER.e_cblp,
            pe.DOS_HEADER.e_cp,
            pe.DOS_HEADER.e_crlc,
            pe.DOS_HEADER.e_cparhdr,
            pe.DOS_HEADER.e_minalloc,
            pe.DOS_HEADER.e_maxalloc,
            pe.DOS_HEADER.e_ss,
            pe.DOS_HEADER.e_sp,
            pe.DOS_HEADER.e_csum,
            pe.DOS_HEADER.e_ip,
            pe.DOS_HEADER.e_cs,
            pe.DOS_HEADER.e_lfarlc,
            pe.DOS_HEADER.e_ovno,
            pe.DOS_HEADER.e_res,
            pe.DOS_HEADER.e_oemid,
            pe.DOS_HEADER.e_oeminfo,
            pe.DOS_HEADER.e_res2,
            pe.DOS_HEADER.e_lfanew,
        ]
        return IMAGE_DOS_HEADER_data

    def extract_features(self, pe):
        IMAGE_DOS_HEADER_data = self.extract_dos_header(pe)

        FILE_HEADER_data = [
            pe.FILE_HEADER.Machine,
            pe.FILE_HEADER.NumberOfSections,
            self.file_creation_year(pe.FILE_HEADER.TimeDateStamp),
            pe.FILE_HEADER.PointerToSymbolTable,
            pe.FILE_HEADER.NumberOfSymbols,
            pe.FILE_HEADER.SizeOfOptionalHeader,
            pe.FILE_HEADER.Characteristics,
        ]

        OPTIONAL_HEADER_data = [
            pe.OPTIONAL_HEADER.Magic,
            pe.OPTIONAL_HEADER.MajorLinkerVersion,
            pe.OPTIONAL_HEADER.MinorLinkerVersion,
            pe.OPTIONAL_HEADER.SizeOfCode,
            pe.OPTIONAL_HEADER.SizeOfInitializedData,
            pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            pe.OPTIONAL_HEADER.BaseOfCode,
            pe.OPTIONAL_HEADER.BaseOfData,
            pe.OPTIONAL_HEADER.ImageBase,
            pe.OPTIONAL_HEADER.SectionAlignment,
            pe.OPTIONAL_HEADER.FileAlignment,
            pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            pe.OPTIONAL_HEADER.MajorImageVersion,
            pe.OPTIONAL_HEADER.MinorImageVersion,
            pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            pe.OPTIONAL_HEADER.SizeOfImage,
            pe.OPTIONAL_HEADER.SizeOfHeaders,
            pe.OPTIONAL_HEADER.CheckSum,
            pe.OPTIONAL_HEADER.Subsystem,
            pe.OPTIONAL_HEADER.DllCharacteristics,
            pe.OPTIONAL_HEADER.SizeOfStackReserve,
            pe.OPTIONAL_HEADER.SizeOfStackCommit,
            pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            pe.OPTIONAL_HEADER.LoaderFlags,
            pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
        ]
        return IMAGE_DOS_HEADER_data + FILE_HEADER_data + OPTIONAL_HEADER_data

    def send_files_recursive(self, rootdir: str = None):
        results = []
        if not rootdir:
            rootdir = self.rootdir
        print("PEAnalyzer scanning: ", rootdir)
        for dirpath, dirs, files in os.walk(rootdir):
            # print(dirpath, dirs, files)
            for filename in files:
                fname = os.path.join(dirpath, filename)
                with open(fname) as myfile:
                    # print(myfile.read())
                    try:
                        pe = pefile.PE(fname)
                        # print("PE: ", fname, pe)
                    except Exception as e:
                        # print("File: ", fname, " cannot be analyzed - ", e)
                        pass
                    else:
                        try:
                            features = self.extract_features(pe)
                            result = (fname, features)
                            results.append(result)
                        except Exception as e:
                            print("Error while saving features: ", e)
                            pass
        return results


if __name__ == "__main__":
    pea = PE_Analyzer()
    pea.send_files_recursive()
