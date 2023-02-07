import json
import os
import sys
from collections import defaultdict

import pydicom
import SimpleITK as sitk
from loguru import logger


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded on the fly"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))


class DicomParser:
    """Filter tag based dicom to nifti converter"""

    def __init__(self, src: str, dst: str, search_tags: dict, log_level: str) -> None:
        self.src = src
        self.dst = dst
        self.search_tags = search_tags
        self.log_level = log_level
        self.path_memory = NestedDefaultDict()
        logger.remove()
        logger.add(sys.stderr, level=self.log_level)

    def __call__(self) -> None:
        logger.info(f'Run: {self.__class__.__name__}')
        self.check_search_tags()
        self.scan_folder()
        self.check_path_memory()
        self.convert_to_nifti()

    def check_search_tags(self):
        """Check if search tags are valid"""
        for sequence, nested in self.search_tags.items():
            for meta_tag in self.search_tags[sequence]['meta_filters'].keys():
                if not isinstance(self.search_tags[sequence]['meta_filters'][meta_tag]['+'], list):
                    raise ValueError(
                        f'Found {type(self.search_tags[sequence]["meta_filters"][meta_tag]["+"])}, expects list'
                    )
            if not isinstance(nested['min_slice_number'], int):
                raise ValueError(f'Found {type(nested["min_slice_number"])}, expects integer')
            if not isinstance(nested['file_extensions'], list):
                raise ValueError(f'Found {type(nested["file_extension"])}, expects list of strings"')

    def show_certain_meta_data(self, tags: list = None, min_slice_number: int = 0) -> None:
        """Iterate and visualise meta data for certain tags"""
        for root, _, files in os.walk(self.src):
            for file in files:
                file_path = os.path.join(root, file)
                if len(os.listdir(os.path.dirname(file_path))) >= min_slice_number:  # slice number
                    print(file_path)
                    ds = pydicom.filereader.dcmread(file_path, force=True)
                    if tags:
                        for tag in tags:
                            print(f'{tag:<20}{ds.get(tag)}')
                    else:
                        print(ds)
                    print('')
                    break

    def check_file_type(self, modality: str, file_name: str) -> bool:
        """True if file ends with defined file type"""
        if [x for x in self.search_tags[modality]['file_extensions'] if file_name.endswith(x)]:
            return True
        return False

    @staticmethod
    def apply_filters(search_values: dict, meta_data: dict) -> bool:
        """Run filters"""
        check = [False]
        if [x for x in search_values['+'] if x in meta_data]:
            check = [True]
        if '-' in search_values:
            if [x for x in search_values['-'] if x in meta_data]:
                check.append(False)
        return all(check)

    def check_tags(self, ds: pydicom.filereader, modality: str, case_name: str) -> bool:
        """Return true in case one of each values for each key has a match"""
        counter = 0
        count_values = 0
        for key in self.search_tags[modality]['meta_filters'].keys():
            meta_data = ds.get(key)
            if meta_data:
                search_values = self.search_tags[modality]['meta_filters'][key]
                count_values += len(search_values['+'])  # sum up multi tag statements
                if self.apply_filters(search_values, meta_data):
                    counter += 1
                    logger.trace(f'{modality} -> {key} -> {search_values} : {meta_data}')
            else:
                raise ValueError(f'Value for case {case_name} with key "{key}" -> None')
        if counter == count_values and counter != 0:
            return True
        return False

    def check_double_findings(self, case_name: str, modality: str, file_path: str) -> None:
        """Helps to resolve double findings"""
        if isinstance(self.path_memory[case_name][modality], str):
            for sequence in self.search_tags.keys():
                ds_1 = pydicom.filereader.dcmread(self.path_memory[case_name][modality])
                ds_2 = pydicom.filereader.dcmread(file_path)
                for search_tag in self.search_tags[sequence]['meta_filters'].keys():
                    logger.error(f'found case -> {ds_1.get(search_tag)}')
                    logger.error(f'found case -> {ds_2.get(search_tag)}')
            raise ValueError(
                f'Modality {modality} got reassigned, add more specific meta_filters,'
                f'\nfile_path_1 -> {self.path_memory[case_name][modality]}'
                f'\nfile_path_2 -> {file_path}'
            )

    def meta_data_search(self, file_path: str) -> None:
        """Check meta data tags"""
        ds = pydicom.filereader.dcmread(file_path, force=True)
        case_name = str(ds.get('PatientName'))
        for modality in self.search_tags:
            if self.check_file(modality, file_path):
                if self.check_tags(ds, modality, case_name):
                    logger.debug(f'found -> {modality} {file_path}')
                    self.check_double_findings(case_name, modality, file_path)
                    self.path_memory[case_name][modality] = file_path

    def check_file(self, modality: str, file_path: str) -> bool:
        """Check found file for certain criteria"""
        check_1, check_2 = False, False
        if os.path.isfile(file_path) and self.check_file_type(modality, file_path):  # exist and file type
            check_1 = True
        count_slices = len(os.listdir(os.path.dirname(file_path)))
        if count_slices >= self.search_tags[modality]['min_slice_number']:  # slice number
            check_2 = True
        return check_1 * check_2

    def scan_folder(self) -> None:
        """Walk through the data set folder and assigns file paths to the nested dict"""
        for root, _, files in os.walk(self.src):
            for file in files:
                file_path = os.path.join(root, file)
                self.meta_data_search(file_path)
                break  # no need to check every file in folder, break out of folder
        logger.info(f'Path memory -> {json.dumps(self.path_memory, indent=4)}')
        logger.info(f'Found unique cases -> {len(self.path_memory)}')

    def check_path_memory(self) -> None:
        """Assures that all sequences for each subject are completed"""
        count_search_tags = len(self.search_tags)
        call_once = True
        for case_name, sequence_names in self.path_memory.items():
            if count_search_tags != len(sequence_names):
                if call_once:
                    call_once = False
                    logger.warning(f'{"Missing data":<20}{"case_name ":<15}sequence')
                missing_sequences = set(self.search_tags).difference(set(sequence_names))
                logger.warning(f'{"":<20}{case_name:<14} {missing_sequences}')

    @staticmethod
    def dicom_sequence_reader(file_path: str) -> sitk.Image:
        """Reads data and meta data of dicom sequences"""
        reader = sitk.ImageSeriesReader()
        file_path = os.path.dirname(file_path)
        series_ids = reader.GetGDCMSeriesIDs(file_path)
        dicom_names = reader.GetGDCMSeriesFileNames(file_path, series_ids[0])
        reader.SetFileNames(dicom_names)
        reader.SetNumberOfThreads(8)
        reader.LoadPrivateTagsOn()
        reader.GlobalWarningDisplayOff()
        img = reader.Execute()
        img = sitk.DICOMOrient(img, 'LPS')
        return img

    def convert_to_nifti(self):
        """Convert path memory to nifti files"""
        for case_name in self.path_memory:
            state = '\u2715'
            for modality in self.path_memory[case_name]:
                try:
                    img = self.dicom_sequence_reader(self.path_memory[case_name][modality])
                    dst_folder = os.path.join(self.dst, case_name)
                    os.makedirs(dst_folder, exist_ok=True)
                    sitk.WriteImage(img, os.path.join(dst_folder, f'{case_name}_{modality}.nii.gz'))
                    state = '\u2713'
                except Exception as error:
                    logger.warning(error)
                finally:
                    logger.info(f'{self.path_memory[case_name][modality]} -> {case_name} -> {state}')


if __name__ == '__main__':
    dp = DicomParser(
        src='/home/melandur/Data/Myocarditis/Boston_Data',
        dst='/home/melandur/Downloads/test_me_you',
        search_tags={
            't1': {
                'meta_filters': {
                    'SeriesDescription': {'+': ['T1', '']},
                    'Modality': {'+': ['MR']},
                    'MRAcquisitionType': {'+': ['2D']},
                },
                'min_slice_number': 2,
                'file_extensions': [''],
            },
        },
        log_level='DEBUG',
    )
    dp()

    dp.show_certain_meta_data(
        [
            'ImageType',
            'Modality',
            'StudyDescription',
            'SeriesNumber',
            'SeriesDescription',
            'MRAcquisitionType',
            'PatientName',
            'SequenceName',
            'ProtocolName',
        ],
        min_slice_number=1,
    )
